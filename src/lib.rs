//! HyperCraft is a VMM crate.

#![no_std]
#![allow(
    clippy::upper_case_acronyms,
    clippy::single_component_path_imports,
    clippy::collapsible_match,
    clippy::default_constructed_unit_structs,
    dead_code,
    non_camel_case_types,
    non_upper_case_globals,
    unused_imports,
    unused_assignments
)]
#![deny(missing_docs, warnings)]
#![feature(
    naked_functions,
    asm_const,
    negative_impls,
    stdsimd,
    inline_const,
    concat_idents
)]

#[macro_use]
extern crate log;
#[macro_use]
extern crate alloc;

#[cfg(target_arch = "aarch64")]
#[path = "arch/aarch64/mod.rs"]
mod arch;
#[cfg(target_arch = "riscv64")]
#[path = "arch/riscv/mod.rs"]
mod arch;
#[cfg(target_arch = "x86_64")]
#[path = "arch/x86_64/mod.rs"]
mod arch;

mod hal;
mod memory;
mod traits;
mod vcpus;

/// HyperCraft Result Define.
pub type HyperResult<T = ()> = Result<T, HyperError>;

#[cfg(not(target_arch = "aarch64"))]
pub use arch::{init_hv_runtime, GprIndex, HyperCallMsg, VmExitInfo};

pub use arch::{NestedPageTable, PerCpu, VCpu, VM};

#[cfg(all(target_arch = "x86_64", feature = "type1_5"))]
pub use arch::LinuxContext;
pub use hal::{HyperCraftHal, MmioOps, PioOps, RegionOps, VirtMsrOps};
#[cfg(target_arch = "x86_64")]
pub use hal::{PerCpuDevices, PerVmDevices};
pub use memory::{
    GuestPageNum, GuestPageTableTrait, GuestPhysAddr, GuestVirtAddr, HostPageNum, HostPhysAddr,
    HostVirtAddr,
};
pub use vcpus::VmCpus;

#[cfg(target_arch = "aarch64")]
pub use arch::lower_aarch64_synchronous;

use alloc::string::String;
#[cfg(target_arch = "x86_64")]
pub use arch::{GuestPageWalkInfo, VmxExitInfo, VmxExitReason};

/// The error type for hypervisor operation failures.
#[derive(Debug, PartialEq)]
pub enum HyperError {
    /// Internal error.
    Internal,
    /// No supported error.
    NotSupported,
    /// No memory error.
    NoMemory,
    /// Invalid parameter error.
    InvalidParam,
    /// Invalid instruction error.
    InvalidInstruction,
    /// Memory out of range error.
    OutOfRange,
    /// Bad state error.
    BadState,
    /// Not found error.
    NotFound,
    /// Fetch instruction error.
    FetchFault,
    /// Page fault error.
    PageFault,
    /// Decode error.
    DecodeError,
    /// Disabled.
    Disabled,
    #[cfg(target_arch = "x86_64")]
    /// Invalid PIO read.
    InValidPioRead,
    #[cfg(target_arch = "x86_64")]
    /// Invalid PIO write.
    InValidPioWrite,
    /// Invalid Mmio read
    InValidMmioRead,
    /// Invalid Mmio write
    InValidMmioWrite,
    /// Pci Error
    PciError(PciError),
    /// Virtio Error
    VirtioError(VirtioError),
}

/// The result type for hypervisor operation.
#[derive(Debug, PartialEq)]
pub enum PciError {
    /// Failed to add PCI capability.
    AddPciCap(u8, usize),
    /// Failed to add PCIe extended capability.
    AddPcieExtCap(u16, usize),
    /// Failed to unmap BAR in memory space.
    UnregMemBar(usize),
    /// Invalid device status.
    DeviceStatus(u32),
    /// Unsupported PCI register.
    PciRegister(u64),
    /// Invalid features select.
    FeaturesSelect(u32),
    /// HotPlug is not supported.
    HotplugUnsupported(u8),
    /// Invalid PCI configuration.
    InvalidConf(String, String),
    /// Failed to enable queue.
    QueueEnable(u32),
    /// Other error.
    Other(String),
}

impl core::fmt::Display for PciError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PciError::AddPciCap(id, size) => write!(
                f,
                "Failed to add PCI capability: id 0x{:x}, size: 0x{:x}.",
                id, size
            ),
            PciError::AddPcieExtCap(id, size) => write!(
                f,
                "Failed to add PCIe extended capability: id 0x{:x}, size: 0x{:x}.",
                id, size
            ),
            PciError::UnregMemBar(index) => {
                write!(f, "Failed to unmap BAR {} in memory space.", index)
            }
            PciError::DeviceStatus(status) => write!(f, "Invalid device status 0x{:x}", status),
            PciError::PciRegister(reg) => write!(f, "Unsupported pci register, 0x{:x}", reg),
            PciError::FeaturesSelect(sel) => write!(f, "Invalid features select 0x{:x}", sel),
            PciError::HotplugUnsupported(devfn) => write!(
                f,
                "HotPlug is not supported for device with devfn {}",
                devfn
            ),
            PciError::InvalidConf(key, value) => {
                write!(f, "Invalid PCI configuration, key:{}, value:{}", key, value)
            }
            PciError::QueueEnable(value) => {
                write!(f, "Failed to enable queue, value is 0x{:x}", value)
            }
            PciError::Other(err) => write!(f, "{}", err),
        }
    }
}

/// Virtio Error Type
#[derive(Debug, PartialEq)]
pub enum VirtioError {
    /// Io error.
    Io {},
    /// EventFd create error.
    EventFdCreate,
    /// EventFd write error.
    EventFdWrite,
    /// Thread create error.
    ThreadCreate(String),
    /// Channel send error.
    ChannelSend(String),
    /// Queue index error.
    QueueIndex(u16, u16),
    /// Queue descriptor invalid error.
    QueueDescInvalid,
    /// Address overflow error.
    AddressOverflow(&'static str, u64, u64),
    /// Device config space overflow error.
    DevConfigOverflow(u64, u64, u64),
    /// Ioctl error.
    VhostIoctl(String),
    /// Element empty error.
    ElementEmpty,
    /// Virt queue is none error.
    VirtQueueIsNone,
    /// Virt queue not enabled error.
    VirtQueueNotEnabled(String, usize),
    /// Incorrect queue number error.
    IncorrectQueueNum(usize, usize),
    /// Incorrect offset error.
    IncorrectOffset(u64, u64),
    /// Device not activated error.
    DeviceNotActivated(String),
    /// Failed to write config error.
    FailedToWriteConfig,
    /// Read object error.
    ReadObjectErr(&'static str, u64),
    /// Device status error.
    DevStatErr(u32),
    /// Mmio register error.
    MmioRegErr(u64),
    /// Other Error
    Other(String),
}

impl core::fmt::Display for VirtioError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VirtioError::Io {} => write!(f, "Io error"),
            VirtioError::EventFdCreate => write!(f, "Failed to create eventfd."),
            VirtioError::EventFdWrite => write!(f, "Failed to write eventfd."),
            VirtioError::ThreadCreate(name) => write!(f, "Failed to create {} thread", name),
            VirtioError::ChannelSend(name) => write!(f, "Failed to send {} on the channel", name),
            VirtioError::QueueIndex(index, size) => {
                write!(f, "Queue index {} invalid, queue size is {}", index, size)
            }
            VirtioError::QueueDescInvalid => write!(f, "Vring descriptor is invalid"),
            VirtioError::AddressOverflow(name, address, offset) => write!(
                f,
                "Address overflows for {}, address: 0x{:x}, offset: {}",
                name, address, offset
            ),
            VirtioError::DevConfigOverflow(offset, len, size) => write!(
                f,
                "Failed to r/w dev config space: overflows, offset {}, len {}, space size {}",
                offset, len, size
            ),
            VirtioError::VhostIoctl(name) => write!(f, "Vhost ioctl failed: {}", name),
            VirtioError::ElementEmpty => write!(f, "Failed to get iovec from element!"),
            VirtioError::VirtQueueIsNone => write!(f, "Virt queue is none!"),
            VirtioError::VirtQueueNotEnabled(dev, queue) => {
                write!(f, "Device {} virt queue {} is not enabled!", dev, queue)
            }
            VirtioError::IncorrectQueueNum(expected, got) => write!(
                f,
                "Cannot perform activate. Expected {} queue(s), got {}",
                expected, got
            ),
            VirtioError::IncorrectOffset(expected, got) => {
                write!(f, "Incorrect offset, expected {}, got {}", expected, got)
            }
            VirtioError::DeviceNotActivated(name) => write!(f, "Device {} not activated", name),
            VirtioError::FailedToWriteConfig => write!(f, "Failed to write config"),
            VirtioError::ReadObjectErr(name, address) => write!(
                f,
                "Failed to read object for {}, address: 0x{:x}",
                name, address
            ),
            VirtioError::DevStatErr(status) => write!(f, "Invalid device status: 0x{:x}.", status),
            VirtioError::MmioRegErr(offset) => {
                write!(f, "Unsupported mmio register at offset 0x{:x}.", offset)
            }
            VirtioError::Other(err) => write!(f, "{}", err),
        }
    }
}
