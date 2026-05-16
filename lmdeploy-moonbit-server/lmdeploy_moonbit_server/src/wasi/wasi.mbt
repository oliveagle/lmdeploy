// src/wasi/wasi.mbt
// WASI (WebAssembly System Interface) bindings for LMDeploy MoonBit API
// This module provides WASI-compatible interfaces for TurboMind FFI calls
// Target: wasm32-unknown-unknown with wasmedge runtime

/// WASI error types
pub enum WasiError {
  Success
  Unknown(Errno)
  NotCapable
  InvalidArgument
  NotFound
  PermissionDenied
  ConnectionRefused
  ConnectionReset
  ConnectionAborted
  AddressNotAvailable
  AddressInUse
  NotConnected
  Again
  TimedOut
  WouldBlock
  Overflow
  NotSupported
  InvalidData
  Io(String)
  Other(String)
}

/// WASI errno values
pub enum Errno {
  Success
  Perm
  NoEnt
  Srch
  Intr
  Io
  Nxio
  TooBig
  NoExec
  BadF
  Child
  Again
  Nomem
  Access
  Fault
  NotEmpty
  NotDir
  IsDir
  Invalid
  Pipe
  Range
  Deadlk
  NotSup
  NoTy
  NoLock
  Exists
  NoSpace
  NotEmpty
  NameTooLong
  Unknown
}

/// WASI file descriptor
pub type Fd = Int

/// WASI file size
pub type FileSize = Int64

/// WASI timestamp (nanoseconds)
pub type Timestamp = Int64

/// WASI directory entry
pub struct DirEntry {
  d_ino: Int64
  d_next: DirCookie
  d_namlen: Int
  d_type: FileType
}

/// WASI directory cookie
pub type DirCookie = Int64

/// WASI file type
pub enum FileType {
  Unknown
  BlockDevice
  CharacterDevice
  Directory
  RegularFile
  SocketDgram
  SocketStream
  SymbolicLink
}

/// WASI I/O vector
pub struct Iovec {
  buf: Array[U8]
  buf_len: Int
}

/// WASI filestat for file metadata
pub struct Filestat {
  dev: Int64
  ino: Int64
  filetype: FileType
  nlink: Int64
  size: FileSize
  atim: Timestamp
  mtim: Timestamp
  ctim: Timestamp
}

/// WASI exit code
pub type ExitCode = Int

/// WASI preopen type
pub enum PreopenType {
  Dir
}

/// WASI prestat for preopened directories
pub struct Prestat {
  type: PreopenType
  u: PrestatDir
}

/// Prestat directory
pub struct PrestatDir {
  pr_name_len: Int
}

/// WASI environment
pub struct WasiEnv {
  args: Array[String]
  env: Array[(String, String)]
  fds: Map[Fd, FdEntry]
  preopened: Map[String, Fd]
}

/// WASI file descriptor entry
pub struct FdEntry {
  fd: Fd
  file_type: FileType
  rights: Rights
  flags: FdFlags
  offset: FileSize
}

/// WASI rights (capabilities)
pub struct Rights {
  base: Int64
  inheriting: Int64
}

/// WASI file descriptor flags
pub struct FdFlags {
  append: Bool
  dsync: Bool
  nonblock: Bool
  rsync: Bool
  sync: Bool
}

/// WASI Fd flags default
pub fn FdFlags::default() -> FdFlags {
  {
    append: false,
    dsync: false,
    nonblock: false,
    rsync: false,
    sync: false,
  }
}

/// WASI open flags
pub struct OpenFlags {
  creat: Bool
  directory: Bool
  excl: Bool
  trunc: Bool
}

/// WASI lookup flags
pub struct LookupFlags {
  follow_symlinks: Bool
}

/// Default WASI environment
pub fn WasiEnv::default() -> WasiEnv {
  {
    args: [],
    env: [],
    fds: Map::new(),
    preopened: Map::new(),
  }
}

/// Initialize WASI environment
pub fn WasiEnv::new() -> WasiEnv {
  let mut env = WasiEnv::default()

  // Preopen standard file descriptors
  // Stdin (0), Stdout (1), Stderr (2)
  let stdin = FdEntry {
    fd: 0,
    file_type: FileType::CharacterDevice,
    rights: Rights { base: 0, inheriting: 0 },
    flags: FdFlags::default(),
    offset: 0,
  }

  let stdout = FdEntry {
    fd: 1,
    file_type: FileType::CharacterDevice,
    rights: Rights { base: 0, inheriting: 0 },
    flags: FdFlags::default(),
    offset: 0,
  }

  let stderr = FdEntry {
    fd: 2,
    file_type: FileType::CharacterDevice,
    rights: Rights { base: 0, inheriting: 0 },
    flags: FdFlags::default(),
    offset: 0,
  }

  let mut fds = Map::new()
  fds = fds.insert(0, stdin)
  fds = fds.insert(1, stdout)
  fds = fds.insert(2, stderr)

  {
    args: env.args,
    env: env.env,
    fds,
    preopened: env.preopened,
  }
}

/// Convert WasiError to string
pub fn WasiError::to_string(self : WasiError) -> String {
  match self {
    Success => "Success"
    Unknown(e) => "Unknown error: " + errno_to_string(e)
    NotCapable => "Not capable"
    InvalidArgument => "Invalid argument"
    NotFound => "Not found"
    PermissionDenied => "Permission denied"
    ConnectionRefused => "Connection refused"
    ConnectionReset => "Connection reset"
    ConnectionAborted => "Connection aborted"
    AddressNotAvailable => "Address not available"
    AddressInUse => "Address in use"
    NotConnected => "Not connected"
    Again => "Resource temporarily unavailable"
    TimedOut => "Operation timed out"
    WouldBlock => "Operation would block"
    Overflow => "Value overflow"
    NotSupported => "Operation not supported"
    InvalidData => "Invalid data"
    Io(s) => "I/O error: " + s
    Other(s) => "Other error: " + s
  }
}

fn errno_to_string(e: Errno) -> String {
  match e {
    Errno::Success => "Success"
    Errno::Perm => "Operation not permitted"
    Errno::NoEnt => "No such file or directory"
    Errno::Srch => "No such process"
    Errno::Intr => "Interrupted system call"
    Errno::Io => "I/O error"
    Errno::Nxio => "No such device or address"
    Errno::TooBig => "File too large"
    Errno::NoExec => "Exec format error"
    Errno::BadF => "Bad file number"
    Errno::Child => "No child processes"
    Errno::Again => "Try again"
    Errno::Nomem => "Out of memory"
    Errno::Access => "Permission denied"
    Errno::Fault => "Bad address"
    Errno::NotEmpty => "Directory not empty"
    Errno::NotDir => "Not a directory"
    Errno::IsDir => "Is a directory"
    Errno::Invalid => "Invalid argument"
    Errno::Pipe => "Broken pipe"
    Errno::Range => "Math result not representable"
    Errno::Deadlk => "Resource deadlock would occur"
    Errno::NotSup => "Not supported"
    Errno::NoTy => "No such device"
    Errno::NoLock => "No record locks available"
    Errno::Exists => "File exists"
    Errno::NoSpace => "No space left on device"
    Errno::NameTooLong => "File name too long"
    Errno::Unknown => "Unknown error"
  }
}
