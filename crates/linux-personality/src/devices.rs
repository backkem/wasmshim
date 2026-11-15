//! Display and input devices: `/dev/hidraw<n>` and `/dev/fb<n>`.
//!
//! **Embedder surface, unstable**: nothing in this repository attaches these devices; the
//! interface follows what embedders need and may change between versions.
//!
//! Both are attached by the embedder before the guest starts ([`Kernel::attach_hid`],
//! [`Kernel::attach_fb`]) and fed or drained between [`Kernel::run`] calls, like the console:
//!
//! * **hidraw.** [`Kernel::hid_input`] hands over one input report exactly as the device sent
//!   it. Every open description has its own queue (Linux's `hidraw_list`, 64 reports; a full
//!   queue drops the new report); `read` returns one report (cut to the buffer), `poll` reports
//!   `EPOLLIN` while one is queued and `EPOLLOUT` always. Output reports (`write`, keyboard
//!   LEDs) are accepted and dropped. The ioctls are the descriptor (`HIDIOCGRDESCSIZE`,
//!   `HIDIOCGRDESC`), `HIDIOCGRAWINFO`, `HIDIOCGRAWNAME`, `HIDIOCGRAWPHYS` and
//!   `HIDIOCGRAWUNIQ`.
//! * **fbdev.** A linear XRGB8888 framebuffer (Linux's `simplefb`/`vfb` model, no panning) in a
//!   memory object, so every `mmap` and `read`/`write` share its bytes.
//!   `FBIOGET_VSCREENINFO`, `FBIOPUT_VSCREENINFO` (the current mode only), `FBIOGET_FSCREENINFO`,
//!   `FBIOPAN_DISPLAY` (offset 0 only), `FBIOBLANK` and `FBIO_WAITFORVSYNC`. At every vertical
//!   blank (vblank `k` at `t0 + ceil(k * den * 1e9 / num)` ns) the kernel
//!   compares the buffer with the frame it showed last; if rows changed it queues an
//!   [`FbFrame`] with the changed rows (the damage), which the embedder takes with
//!   [`Kernel::take_frames`] and presents.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::{Rc, Weak};

use wasmshim_linux_abi::Errno;

use crate::fd::FileKind;
use crate::kernel::{Kernel, Tid, WaitKey};
use crate::objects::{EPOLLIN, EPOLLOUT, EPOLLRDNORM, EPOLLWRNORM};
use crate::shm::MemRef;
use crate::syscall::SysResult;
use crate::vfs::Device;

// ---- hidraw ---------------------------------------------------------------------------------

/// Reports a description queues before new ones are dropped (`HIDRAW_BUFFER_SIZE`).
const HID_QUEUE: usize = 64;
/// `HID_MAX_DESCRIPTOR_SIZE`.
const HID_MAX_DESCRIPTOR: usize = 4096;
/// `BUS_USB`.
pub const BUS_USB: u32 = 0x03;

/// The boot keyboard's report descriptor (HID 1.11, appendix B.1): modifiers, a reserved
/// byte, six key codes; five LED output bits.
pub const BOOT_KEYBOARD_DESCRIPTOR: &[u8] = &[
    0x05, 0x01, 0x09, 0x06, 0xa1, 0x01, 0x05, 0x07, 0x19, 0xe0, 0x29, 0xe7, 0x15, 0x00, 0x25, 0x01,
    0x75, 0x01, 0x95, 0x08, 0x81, 0x02, 0x95, 0x01, 0x75, 0x08, 0x81, 0x01, 0x95, 0x05, 0x75, 0x01,
    0x05, 0x08, 0x19, 0x01, 0x29, 0x05, 0x91, 0x02, 0x95, 0x01, 0x75, 0x03, 0x91, 0x01, 0x95, 0x06,
    0x75, 0x08, 0x15, 0x00, 0x25, 0x65, 0x05, 0x07, 0x19, 0x00, 0x29, 0x65, 0x81, 0x00, 0xc0,
];

/// The boot mouse's report descriptor (HID 1.11, appendix B.2) with a wheel byte: three
/// buttons, relative X and Y, and the wheel, each a signed byte.
pub const BOOT_MOUSE_DESCRIPTOR: &[u8] = &[
    0x05, 0x01, 0x09, 0x02, 0xa1, 0x01, 0x09, 0x01, 0xa1, 0x00, 0x05, 0x09, 0x19, 0x01, 0x29, 0x03,
    0x15, 0x00, 0x25, 0x01, 0x95, 0x03, 0x75, 0x01, 0x81, 0x02, 0x95, 0x01, 0x75, 0x05, 0x81, 0x01,
    0x05, 0x01, 0x09, 0x30, 0x09, 0x31, 0x09, 0x38, 0x15, 0x81, 0x25, 0x7f, 0x75, 0x08, 0x95, 0x03,
    0x81, 0x06, 0xc0, 0xc0,
];

/// What the embedder says about a HID device plugged into the node (there is no USB
/// enumeration: the descriptor comes with it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HidInfo {
    /// `HIDIOCGRAWNAME`.
    pub name: String,
    /// `HIDIOCGRAWPHYS`: where it is plugged in.
    pub phys: String,
    /// `HIDIOCGRAWINFO`.
    pub bus: u32,
    pub vendor: u16,
    pub product: u16,
    /// The report descriptor (at most 4096 bytes).
    pub descriptor: Vec<u8>,
}

impl HidInfo {
    /// A boot-protocol keyboard (8-byte reports).
    pub fn boot_keyboard(name: &str, phys: &str) -> Self {
        Self {
            name: name.into(),
            phys: phys.into(),
            bus: BUS_USB,
            vendor: 0x1209,
            product: 0x0001,
            descriptor: BOOT_KEYBOARD_DESCRIPTOR.to_vec(),
        }
    }

    /// A boot-protocol mouse with a wheel (4-byte reports).
    pub fn boot_mouse(name: &str, phys: &str) -> Self {
        Self {
            name: name.into(),
            phys: phys.into(),
            bus: BUS_USB,
            vendor: 0x1209,
            product: 0x0002,
            descriptor: BOOT_MOUSE_DESCRIPTOR.to_vec(),
        }
    }
}

/// One attached HID device.
#[derive(Debug)]
pub(crate) struct HidDevice {
    pub(crate) info: HidInfo,
    pub(crate) opens: Vec<Weak<RefCell<HidOpen>>>,
}

/// One open description of `/dev/hidraw<dev>`: its own report queue.
#[derive(Debug)]
pub(crate) struct HidOpen {
    pub id: u64,
    pub dev: u32,
    /// The inode it was opened through (`fstat`).
    pub ino: crate::vfs::Ino,
    pub(crate) queue: VecDeque<Vec<u8>>,
    pub(crate) generation: u64,
}

impl HidOpen {
    pub(crate) fn events(&self) -> (u32, u64) {
        let mut ev = EPOLLOUT | EPOLLWRNORM;
        if !self.queue.is_empty() {
            ev |= EPOLLIN | EPOLLRDNORM;
        }
        (ev, self.generation)
    }
}

pub(crate) type HidRef = Rc<RefCell<HidOpen>>;

// ---- fbdev ----------------------------------------------------------------------------------

/// `FB_TYPE_PACKED_PIXELS`, `FB_VISUAL_TRUECOLOR`.
const FB_TYPE_PACKED_PIXELS: u32 = 0;
const FB_VISUAL_TRUECOLOR: u32 = 2;
/// `FB_BLANK_UNBLANK` .. `FB_BLANK_POWERDOWN`.
const FB_BLANK_POWERDOWN: u32 = 4;
/// `sizeof(struct fb_var_screeninfo)`, `sizeof(struct fb_fix_screeninfo)` on 64-bit.
pub const FB_VAR_SIZE: usize = 160;
pub const FB_FIX_SIZE: usize = 80;
/// DRM's `XR24` (`DRM_FORMAT_XRGB8888`): 32 bits, B G R X in memory.
pub const FOURCC_XRGB8888: u32 = 0x3432_5258;

/// A framebuffer's mode, fixed at attach time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FbMode {
    pub width: u32,
    pub height: u32,
    /// Refresh rate as a ratio (Hz = num / den).
    pub refresh_num: u32,
    pub refresh_den: u32,
}

/// A frame taken at a vertical blank: rows `y .. y + height` changed (full width, XRGB8888,
/// row-major without padding: `pixels.len() == width * height * 4`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FbFrame {
    /// `/dev/fb<index>`.
    pub index: u32,
    /// The vblank's virtual time (ns).
    pub at: u64,
    /// The vblank's number since attach (0 is the first).
    pub vblank: u64,
    pub y: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct Framebuffer {
    pub(crate) id: u64,
    pub(crate) mode: FbMode,
    /// Bytes per row (`line_length`).
    pub(crate) stride: u32,
    /// `smem_len`: the rows, rounded up to a page.
    pub(crate) size: u64,
    pub(crate) mem: MemRef,
    /// What the screen shows (the last frame taken).
    pub(crate) shown: Vec<u8>,
    pub(crate) scratch: Vec<u8>,
    pub(crate) blank: bool,
    /// Compare every row at the next vblank (a read of the screen memory failed).
    pub(crate) full_next: bool,
    pub(crate) t0: u64,
    /// The next vblank's number.
    pub(crate) next: u64,
}

/// The rows (ascending, merged ranges) of a `rows`-row screen with `row` bytes per row that
/// lie on pages `pages` (ascending page indices) of its memory.
fn dirty_rows(pages: &[u64], row: usize, rows: usize) -> Vec<core::ops::Range<usize>> {
    let total = row * rows;
    let mut out: Vec<core::ops::Range<usize>> = Vec::new();
    for &p in pages {
        let start = (p as usize).saturating_mul(4096);
        if start >= total {
            break;
        }
        let end = start.saturating_add(4096).min(total);
        let r = start / row..end.div_ceil(row);
        match out.last_mut() {
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => out.push(r),
        }
    }
    out
}

impl Framebuffer {
    fn vblank_time(&self, k: u64) -> u64 {
        let (num, den) = (
            u128::from(self.mode.refresh_num.max(1)),
            u128::from(self.mode.refresh_den.max(1)),
        );
        let ns = (u128::from(k) * den * 1_000_000_000).div_ceil(num);
        self.t0.saturating_add(ns.min(u128::from(u64::MAX)) as u64)
    }

    /// `struct fb_var_screeninfo` for the current mode.
    fn var(&self) -> [u8; FB_VAR_SIZE] {
        let m = self.mode;
        let mut v = [0u8; FB_VAR_SIZE];
        let mut put = |off: usize, x: u32| v[off..off + 4].copy_from_slice(&x.to_le_bytes());
        put(0, m.width); // xres
        put(4, m.height); // yres
        put(8, m.width); // xres_virtual
        put(12, m.height); // yres_virtual
        put(24, 32); // bits_per_pixel
        put(32, 16); // red.offset
        put(36, 8); // red.length
        put(44, 8); // green.offset
        put(48, 8); // green.length
        put(56, 0); // blue.offset
        put(60, 8); // blue.length
        put(88, u32::MAX); // height (mm): unknown
        put(92, u32::MAX); // width (mm): unknown
        // pixclock (ps per pixel) of a mode with no blanking intervals.
        let pixels = u128::from(m.width) * u128::from(m.height);
        let hz_num = u128::from(m.refresh_num.max(1));
        let hz_den = u128::from(m.refresh_den.max(1));
        let pixclock = if pixels > 0 {
            (1_000_000_000_000u128 * hz_den / (hz_num * pixels)).min(u128::from(u32::MAX)) as u32
        } else {
            0
        };
        put(100, pixclock);
        v
    }

    /// `struct fb_fix_screeninfo`.
    fn fix(&self, index: u32, prefix: &str) -> [u8; FB_FIX_SIZE] {
        let mut f = [0u8; FB_FIX_SIZE];
        let id = format!("{prefix}{index}");
        f[..id.len().min(15)].copy_from_slice(&id.as_bytes()[..id.len().min(15)]);
        // smem_start (16..24) stays 0: no physical address to give.
        f[24..28].copy_from_slice(&(self.size as u32).to_le_bytes());
        f[28..32].copy_from_slice(&FB_TYPE_PACKED_PIXELS.to_le_bytes());
        f[36..40].copy_from_slice(&FB_VISUAL_TRUECOLOR.to_le_bytes());
        f[48..52].copy_from_slice(&self.stride.to_le_bytes());
        f
    }
}

// ---- ioctl numbers --------------------------------------------------------------------------

mod nr {
    pub const FBIOGET_VSCREENINFO: u64 = 0x4600;
    pub const FBIOPUT_VSCREENINFO: u64 = 0x4601;
    pub const FBIOGET_FSCREENINFO: u64 = 0x4602;
    pub const FBIOPAN_DISPLAY: u64 = 0x4606;
    pub const FBIOBLANK: u64 = 0x4611;
    /// `_IOW('F', 0x20, __u32)`.
    pub const FBIO_WAITFORVSYNC: u64 = 0x4004_4620;
    /// `_IOR('H', 0x01, int)`.
    pub const HIDIOCGRDESCSIZE: u64 = 0x8004_4801;
    /// `_IOR('H', 0x02, struct hidraw_report_descriptor)`.
    pub const HIDIOCGRDESC: u64 = 0x9004_4802;
    /// `_IOR('H', 0x03, struct hidraw_devinfo)`.
    pub const HIDIOCGRAWINFO: u64 = 0x8008_4803;
}

/// `_IOC_NR`, `_IOC_TYPE`, `_IOC_SIZE`, `_IOC_DIR` of a request.
fn ioc(req: u64) -> (u64, u64, usize, u64) {
    (
        req & 0xff,
        (req >> 8) & 0xff,
        ((req >> 16) & 0x3fff) as usize,
        req >> 30,
    )
}

impl Kernel {
    fn mem_of(&self, tid: Tid) -> Rc<RefCell<crate::mm::AddressSpace>> {
        self.mm(tid)
    }

    // ---- hidraw ----

    /// Plugs a HID device in as `/dev/hidraw<n>` (`n` counts from 0 in attach order).
    pub fn attach_hid(&mut self, info: HidInfo) -> u32 {
        let n = self.hid.len() as u32;
        self.vfs
            .add_device(format!("hidraw{n}").as_bytes(), Device::HidRaw(n), 0o600);
        self.hid.push(HidDevice {
            info,
            opens: Vec::new(),
        });
        n
    }

    /// The attached HID devices, in `/dev/hidraw<n>` order.
    pub fn hid_devices(&self) -> Vec<HidInfo> {
        self.hid.iter().map(|d| d.info.clone()).collect()
    }

    /// One input report from HID device `dev`, as it came off the device: queued on every open
    /// description of `/dev/hidraw<dev>` (none open: dropped, as on Linux).
    pub fn hid_input(&mut self, dev: u32, report: &[u8]) {
        let Some(d) = self.hid.get_mut(dev as usize) else {
            return;
        };
        d.opens.retain(|w| w.strong_count() > 0);
        let mut changed = Vec::new();
        for open in d.opens.iter().filter_map(Weak::upgrade) {
            let mut o = open.borrow_mut();
            if o.queue.len() < HID_QUEUE {
                o.queue.push_back(report.to_vec());
                o.generation += 1;
                changed.push(o.id);
            }
        }
        for id in changed {
            self.object_changed(id);
        }
    }

    pub(crate) fn hidraw_open(
        &mut self,
        dev: u32,
        ino: crate::vfs::Ino,
    ) -> Result<FileKind, Errno> {
        let id = self.new_object_id();
        let d = self.hid.get_mut(dev as usize).ok_or(Errno::ENODEV)?;
        let open = Rc::new(RefCell::new(HidOpen {
            id,
            dev,
            ino,
            queue: VecDeque::new(),
            generation: 0,
        }));
        d.opens.push(Rc::downgrade(&open));
        Ok(FileKind::HidRaw(open))
    }

    pub(crate) fn hidraw_read(
        &mut self,
        h: &HidRef,
        buf: &mut [u8],
        nonblock: bool,
    ) -> Result<usize, Errno> {
        let mut o = h.borrow_mut();
        let Some(report) = o.queue.pop_front() else {
            if nonblock {
                return Err(Errno::EAGAIN);
            }
            let id = o.id;
            drop(o);
            return Err(self.wait_for(WaitKey::Object(id), None, None));
        };
        o.generation += 1;
        let n = report.len().min(buf.len());
        buf[..n].copy_from_slice(&report[..n]);
        Ok(n)
    }

    pub(crate) fn hidraw_ioctl(&mut self, tid: Tid, h: &HidRef, req: u64, arg: u64) -> SysResult {
        let dev = h.borrow().dev;
        let info = self
            .hid
            .get(dev as usize)
            .map(|d| d.info.clone())
            .ok_or(Errno::ENODEV)?;
        let (number, ty, size, dir) = ioc(req);
        // As hidraw_ioctl: another driver's request is EINVAL, an unknown hidraw one ENOTTY.
        if ty != u64::from(b'H') {
            return Err(Errno::EINVAL);
        }
        let mm = self.mem_of(tid);
        let mut mm = mm.borrow_mut();
        let mem = mm.mem.as_mut();
        match req {
            nr::HIDIOCGRDESCSIZE => {
                crate::uaccess::write_u32(mem, arg, info.descriptor.len() as u32)?;
                Ok(0)
            }
            nr::HIDIOCGRDESC => {
                let want = crate::uaccess::read_array::<4>(mem, arg)?;
                let want = u32::from_le_bytes(want) as usize;
                if want > HID_MAX_DESCRIPTOR - 1 {
                    return Err(Errno::EINVAL);
                }
                let n = want.min(info.descriptor.len());
                crate::uaccess::write_bytes(mem, arg + 4, &info.descriptor[..n])?;
                Ok(0)
            }
            nr::HIDIOCGRAWINFO => {
                let mut b = [0u8; 8];
                b[..4].copy_from_slice(&info.bus.to_le_bytes());
                b[4..6].copy_from_slice(&info.vendor.to_le_bytes());
                b[6..8].copy_from_slice(&info.product.to_le_bytes());
                crate::uaccess::write_bytes(mem, arg, &b)?;
                Ok(0)
            }
            // HIDIOCGRAWNAME(len), HIDIOCGRAWPHYS(len), HIDIOCGRAWUNIQ(len): the string with
            // its NUL, cut to `len`; the result is the length copied.
            _ if dir == 2 && matches!(number, 0x04 | 0x05 | 0x08) => {
                let s = match number {
                    0x04 => info.name.as_bytes(),
                    0x05 => info.phys.as_bytes(),
                    _ => b"",
                };
                let mut bytes = s.to_vec();
                bytes.push(0);
                bytes.truncate(size);
                crate::uaccess::write_bytes(mem, arg, &bytes)?;
                Ok(bytes.len() as u64)
            }
            _ => Err(Errno::ENOTTY),
        }
    }

    // ---- fbdev ----

    /// Plugs in a framebuffer as `/dev/fb<n>`; its vblanks count from `now`.
    pub fn attach_fb(&mut self, mode: FbMode, now: u64) -> Result<u32, Errno> {
        if mode.width == 0 || mode.height == 0 || mode.width > 8192 || mode.height > 8192 {
            return Err(Errno::EINVAL);
        }
        let n = self.fbs.len() as u32;
        let stride = mode.width * 4;
        let bytes = u64::from(stride) * u64::from(mode.height);
        let size = bytes.div_ceil(4096) * 4096;
        let mem = self.new_mem_object(None, Some(size))?;
        let id = self.new_object_id();
        self.vfs
            .add_device(format!("fb{n}").as_bytes(), Device::Fb(n), 0o660);
        self.fbs.push(Framebuffer {
            id,
            mode,
            stride,
            size,
            mem,
            shown: vec![0; bytes as usize],
            scratch: vec![0; bytes as usize],
            blank: false,
            full_next: false,
            t0: now,
            next: 1,
        });
        Ok(n)
    }

    /// The attached framebuffers' modes, in `/dev/fb<n>` order.
    pub fn fb_modes(&self) -> Vec<FbMode> {
        self.fbs.iter().map(|f| f.mode).collect()
    }

    /// Frames taken at vblanks since the last call, oldest first.
    pub fn take_frames(&mut self) -> Vec<FbFrame> {
        std::mem::take(&mut self.fb_frames)
    }

    /// What every framebuffer shows now, whole: one full-screen frame per framebuffer that is
    /// not blanked, stamped `now`. For an embedder whose display starts empty while the kernel's
    /// screens do not, such as a machine restored from a checkpoint: vblanks only
    /// send the rows that change, so it presents these once. Takes nothing and changes nothing.
    pub fn fb_screens(&self, now: u64) -> Vec<FbFrame> {
        self.fbs
            .iter()
            .enumerate()
            .filter(|(_, fb)| !fb.blank)
            .map(|(index, fb)| FbFrame {
                index: index as u32,
                at: now,
                vblank: fb.next.saturating_sub(1),
                y: 0,
                height: fb.mode.height,
                pixels: fb.shown.clone(),
            })
            .collect()
    }

    /// The earliest next vblank (a kernel deadline while a framebuffer is attached).
    pub(crate) fn next_vblank(&self) -> Option<u64> {
        self.fbs.iter().map(|f| f.vblank_time(f.next)).min()
    }

    /// Takes frames for every vblank due at `now` (one frame per framebuffer however many
    /// vblanks passed: a busy machine misses vblanks, as a real one would drop frames).
    pub(crate) fn expire_vblanks(&mut self, now: u64) {
        for (index, fb) in self.fbs.iter_mut().enumerate() {
            let due = fb.vblank_time(fb.next);
            if due > now {
                continue;
            }
            let k = fb.next;
            // The first vblank after `now`.
            while fb.vblank_time(fb.next) <= now {
                fb.next += 1;
            }
            if fb.blank {
                continue;
            }
            let row = fb.stride as usize;
            let rows = fb.mode.height as usize;
            // `shown` equals the screen memory except where it was stored to since: compare
            // only the rows on those pages (every row when the object cannot tell).
            let dirty = fb.mem.pages.take_dirty();
            let checked = match dirty {
                Some(pages) if !fb.full_next => dirty_rows(&pages, row, rows),
                _ => core::iter::once(0..rows).collect(),
            };
            let read = checked.iter().try_for_each(|r| {
                fb.mem.pages.read(
                    (r.start * row) as u64,
                    &mut fb.scratch[r.start * row..r.end * row],
                )
            });
            // A failed read loses what was stored: compare everything next time.
            fb.full_next = read.is_err();
            if read.is_err() {
                continue;
            }
            let same =
                |y: usize| fb.scratch[y * row..(y + 1) * row] == fb.shown[y * row..(y + 1) * row];
            let Some(first) = checked.iter().flat_map(|r| r.clone()).find(|&y| !same(y)) else {
                continue;
            };
            let last = checked
                .iter()
                .rev()
                .flat_map(|r| r.clone().rev())
                .find(|&y| !same(y))
                .unwrap_or(first);
            // Rows of the span that were not checked are unchanged: `shown` has them.
            for r in &checked {
                let (a, b) = (r.start.max(first), r.end.min(last + 1));
                if a < b {
                    fb.shown[a * row..b * row].copy_from_slice(&fb.scratch[a * row..b * row]);
                }
            }
            let span = first * row..(last + 1) * row;
            self.fb_frames.push(FbFrame {
                index: index as u32,
                at: due,
                vblank: k - 1,
                y: first as u32,
                height: (last + 1 - first) as u32,
                pixels: fb.shown[span].to_vec(),
            });
        }
    }

    fn fb(&self, n: u32) -> Result<&Framebuffer, Errno> {
        self.fbs.get(n as usize).ok_or(Errno::ENODEV)
    }

    /// The memory object a shared mapping of `/dev/fb<n>` maps, and its size.
    pub(crate) fn fb_object(&self, n: u32) -> Result<(MemRef, u64), Errno> {
        let fb = self.fb(n)?;
        Ok((fb.mem.clone(), fb.size))
    }

    pub(crate) fn fb_size(&self, n: u32) -> u64 {
        self.fb(n).map_or(0, |f| f.size)
    }

    /// `read`/`pread` of `/dev/fb<n>` at `pos`: the screen memory (end of file at its size).
    pub(crate) fn fb_read(&self, n: u32, pos: u64, buf: &mut [u8]) -> Result<usize, Errno> {
        let fb = self.fb(n)?;
        if pos >= fb.size {
            return Ok(0);
        }
        let len = (buf.len() as u64).min(fb.size - pos) as usize;
        fb.mem
            .pages
            .read(pos, &mut buf[..len])
            .map_err(|_| Errno::EIO)?;
        Ok(len)
    }

    /// `write`/`pwrite` of `/dev/fb<n>`: past the end is `ENOSPC` (`EFBIG` at the end, as
    /// `fb_write` does).
    pub(crate) fn fb_write(&self, n: u32, pos: u64, data: &[u8]) -> Result<usize, Errno> {
        let fb = self.fb(n)?;
        if pos > fb.size {
            return Err(Errno::EFBIG);
        }
        let len = (data.len() as u64).min(fb.size - pos) as usize;
        if len == 0 && !data.is_empty() {
            return Err(Errno::ENOSPC);
        }
        fb.mem
            .pages
            .write(pos, &data[..len])
            .map_err(|_| Errno::EIO)?;
        Ok(len)
    }

    pub(crate) fn fb_ioctl(&mut self, tid: Tid, n: u32, req: u64, arg: u64) -> SysResult {
        let (var, fix, id, next) = {
            let fb = self.fb(n)?;
            (
                fb.var(),
                fb.fix(n, &self.cfg.fb_id),
                fb.id,
                fb.vblank_time(fb.next),
            )
        };
        let mm = self.mem_of(tid);
        match req {
            nr::FBIOGET_VSCREENINFO => {
                crate::uaccess::write_bytes(mm.borrow_mut().mem.as_mut(), arg, &var)?;
                Ok(0)
            }
            nr::FBIOGET_FSCREENINFO => {
                crate::uaccess::write_bytes(mm.borrow_mut().mem.as_mut(), arg, &fix)?;
                Ok(0)
            }
            nr::FBIOPUT_VSCREENINFO => {
                // Only the current mode: the geometry and depth must match; the rest of the
                // structure is written back as the device has it.
                let want: [u8; FB_VAR_SIZE] =
                    crate::uaccess::read_array(mm.borrow_mut().mem.as_mut(), arg)?;
                if want[..16] != var[..16] || want[24..28] != var[24..28] {
                    return Err(Errno::EINVAL);
                }
                crate::uaccess::write_bytes(mm.borrow_mut().mem.as_mut(), arg, &var)?;
                Ok(0)
            }
            nr::FBIOPAN_DISPLAY => {
                let want: [u8; FB_VAR_SIZE] =
                    crate::uaccess::read_array(mm.borrow_mut().mem.as_mut(), arg)?;
                if want[16..24] != [0; 8] {
                    return Err(Errno::EINVAL);
                }
                Ok(0)
            }
            nr::FBIOBLANK => {
                if arg > u64::from(FB_BLANK_POWERDOWN) {
                    return Err(Errno::EINVAL);
                }
                if let Some(fb) = self.fbs.get_mut(n as usize) {
                    fb.blank = arg != 0;
                }
                Ok(0)
            }
            // Returns at the next vblank (the wait times out there with result 0).
            nr::FBIO_WAITFORVSYNC => Err(self.wait_for(WaitKey::Object(id), Some(next), Some(0))),
            _ => Err(Errno::ENOTTY),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boot_descriptors_classify() {
        assert_eq!(&BOOT_KEYBOARD_DESCRIPTOR[..4], &[0x05, 0x01, 0x09, 0x06]);
        assert_eq!(&BOOT_MOUSE_DESCRIPTOR[..4], &[0x05, 0x01, 0x09, 0x02]);
        assert_eq!(*BOOT_KEYBOARD_DESCRIPTOR.last().unwrap(), 0xc0);
    }

    #[test]
    fn ioctl_numbers_decode() {
        assert_eq!(ioc(nr::HIDIOCGRDESC), (2, 0x48, 4100, 2));
        assert_eq!(ioc(nr::HIDIOCGRAWINFO), (3, 0x48, 8, 2));
        assert_eq!(ioc(nr::FBIO_WAITFORVSYNC), (0x20, 0x46, 4, 1));
    }

    #[test]
    fn dirty_pages_to_rows() {
        let one = |r: core::ops::Range<usize>| vec![r];
        // 1024 x 768 XRGB: one row per page.
        assert_eq!(
            dirty_rows(&[0, 1, 5, 767, 768], 4096, 768),
            [0..2, 5..6, 767..768]
        );
        // 100-pixel rows (400 bytes): page 1 holds bytes 4096..8192, rows 10..21.
        assert_eq!(dirty_rows(&[1], 400, 50), one(10..21));
        assert_eq!(dirty_rows(&[0, 1], 400, 50), one(0..21));
        // The last page is partly past the rows.
        assert_eq!(dirty_rows(&[4], 400, 50), one(40..50));
        assert!(dirty_rows(&[], 400, 50).is_empty());
    }
}
