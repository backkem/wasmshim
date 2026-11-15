//! Credentials: real/effective/saved/filesystem user and group ids and supplementary groups.
//!
//! Tracked per process (musl and glibc apply `setuid` and friends to every thread anyway) and
//! reported faithfully (`getuid`, `/proc/<pid>/status`, `capget`), but **not enforced**: file
//! permissions are never checked, so a process that drops root (nginx's workers
//! run as `nginx`) keeps working on the in-memory overlay. Changing ids follows Linux's rules
//! for privileged and unprivileged callers, so programs that check for `EPERM` see it.

use wasmshim_linux_abi::Errno;

use super::SysResult;
use crate::kernel::{Kernel, Tid};
use crate::uaccess::{write_bytes, write_u32};

/// `NGROUPS_MAX`.
const NGROUPS_MAX: u64 = 65536;

/// One process's ids.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Creds {
    pub ruid: u32,
    pub euid: u32,
    pub suid: u32,
    pub fsuid: u32,
    pub rgid: u32,
    pub egid: u32,
    pub sgid: u32,
    pub fsgid: u32,
    pub groups: Vec<u32>,
}

impl Creds {
    /// Has `CAP_SETUID`/`CAP_SETGID`: root keeps its capabilities while its effective uid is 0.
    pub fn privileged(&self) -> bool {
        self.euid == 0
    }
}

/// `-1` (as `u32`) leaves an id unchanged in the `setre*`/`setres*` calls.
const KEEP: u32 = u32::MAX;

impl Kernel {
    fn creds_mut(&mut self, tid: Tid) -> &mut Creds {
        let pid = self.pid_of(tid);
        &mut self.procs.get_mut(&pid).expect("caller's process").creds
    }

    /// The caller's credentials.
    pub(crate) fn creds(&self, tid: Tid) -> &Creds {
        &self.procs[&self.pid_of(tid)].creds
    }

    pub(super) fn sys_getid(&mut self, tid: Tid, which: wasmshim_linux_abi::Sysno) -> SysResult {
        use wasmshim_linux_abi::Sysno;
        let c = self.creds(tid);
        Ok(u64::from(match which {
            Sysno::getuid => c.ruid,
            Sysno::geteuid => c.euid,
            Sysno::getgid => c.rgid,
            _ => c.egid,
        }))
    }

    pub(super) fn sys_getresid(
        &mut self,
        tid: Tid,
        a: u64,
        b: u64,
        c: u64,
        group: bool,
    ) -> SysResult {
        let cr = self.creds(tid).clone();
        let ids = if group {
            [cr.rgid, cr.egid, cr.sgid]
        } else {
            [cr.ruid, cr.euid, cr.suid]
        };
        let mm = self.mm(tid);
        let mut mm = mm.borrow_mut();
        for (p, id) in [a, b, c].into_iter().zip(ids) {
            write_u32(mm.mem.as_mut(), p, id)?;
        }
        Ok(0)
    }

    /// `setuid`/`setgid`: privileged callers set all three ids, others only the effective
    /// one, and only to their real or saved id.
    pub(super) fn sys_setid(&mut self, tid: Tid, id: u64, group: bool) -> SysResult {
        let id = id as u32;
        if id == KEEP {
            return Err(Errno::EINVAL);
        }
        let c = self.creds_mut(tid);
        let privileged = c.privileged();
        let (r, e, s, fs) = if group {
            (&mut c.rgid, &mut c.egid, &mut c.sgid, &mut c.fsgid)
        } else {
            (&mut c.ruid, &mut c.euid, &mut c.suid, &mut c.fsuid)
        };
        if privileged {
            (*r, *e, *s) = (id, id, id);
        } else if id == *r || id == *s {
            *e = id;
        } else {
            return Err(Errno::EPERM);
        }
        *fs = *e;
        Ok(0)
    }

    /// `setreuid`/`setregid`.
    pub(super) fn sys_setreid(&mut self, tid: Tid, ruid: u64, euid: u64, group: bool) -> SysResult {
        let (nr, ne) = (ruid as u32, euid as u32);
        let c = self.creds_mut(tid);
        let privileged = c.privileged();
        let (r, e, s, fs) = if group {
            (&mut c.rgid, &mut c.egid, &mut c.sgid, &mut c.fsgid)
        } else {
            (&mut c.ruid, &mut c.euid, &mut c.suid, &mut c.fsuid)
        };
        if !privileged
            && ((nr != KEEP && nr != *r && nr != *e)
                || (ne != KEEP && ne != *r && ne != *e && ne != *s))
        {
            return Err(Errno::EPERM);
        }
        let old_r = *r;
        if nr != KEEP {
            *r = nr;
        }
        if ne != KEEP {
            *e = ne;
        }
        // The saved id follows the new effective id when the real id is set, or the effective
        // id changes to something other than the old real id.
        if nr != KEEP || (ne != KEEP && ne != old_r) {
            *s = *e;
        }
        *fs = *e;
        Ok(0)
    }

    /// `setresuid`/`setresgid`.
    pub(super) fn sys_setresid(
        &mut self,
        tid: Tid,
        nr: u64,
        ne: u64,
        ns: u64,
        group: bool,
    ) -> SysResult {
        let new = [nr as u32, ne as u32, ns as u32];
        let c = self.creds_mut(tid);
        let privileged = c.privileged();
        let (r, e, s, fs) = if group {
            (&mut c.rgid, &mut c.egid, &mut c.sgid, &mut c.fsgid)
        } else {
            (&mut c.ruid, &mut c.euid, &mut c.suid, &mut c.fsuid)
        };
        let allowed = [*r, *e, *s];
        if !privileged && new.iter().any(|&id| id != KEEP && !allowed.contains(&id)) {
            return Err(Errno::EPERM);
        }
        if new[0] != KEEP {
            *r = new[0];
        }
        if new[1] != KEEP {
            *e = new[1];
        }
        if new[2] != KEEP {
            *s = new[2];
        }
        *fs = *e;
        Ok(0)
    }

    /// `setfsuid`/`setfsgid`: always returns the previous value.
    pub(super) fn sys_setfsid(&mut self, tid: Tid, id: u64, group: bool) -> SysResult {
        let id = id as u32;
        let c = self.creds_mut(tid);
        let privileged = c.privileged();
        let (r, e, s, fs) = if group {
            (c.rgid, c.egid, c.sgid, &mut c.fsgid)
        } else {
            (c.ruid, c.euid, c.suid, &mut c.fsuid)
        };
        let old = *fs;
        if id != KEEP && (privileged || [r, e, s, old].contains(&id)) {
            *fs = id;
        }
        Ok(u64::from(old))
    }

    pub(super) fn sys_getgroups(&mut self, tid: Tid, size: u64, list: u64) -> SysResult {
        let groups = self.creds(tid).groups.clone();
        if size == 0 {
            return Ok(groups.len() as u64);
        }
        if (size as usize) < groups.len() {
            return Err(Errno::EINVAL);
        }
        let bytes: Vec<u8> = groups.iter().flat_map(|g| g.to_le_bytes()).collect();
        write_bytes(self.mm(tid).borrow_mut().mem.as_mut(), list, &bytes)?;
        Ok(groups.len() as u64)
    }

    pub(super) fn sys_setgroups(&mut self, tid: Tid, size: u64, list: u64) -> SysResult {
        if size > NGROUPS_MAX {
            return Err(Errno::EINVAL);
        }
        if !self.creds(tid).privileged() {
            return Err(Errno::EPERM);
        }
        let bytes = crate::uaccess::read_bytes(
            self.mm(tid).borrow_mut().mem.as_mut(),
            list,
            size as usize * 4,
        )?;
        let groups = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        self.creds_mut(tid).groups = groups;
        Ok(0)
    }
}
