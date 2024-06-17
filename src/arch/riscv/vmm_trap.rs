#[derive(Debug, Clone, Copy)]
/// Identifies the reason for a trap taken from a vCPU.
pub enum VmmTrap {
    /// 設定 timer
    SetTimer(u64),
    /// An timer interrupt for the running vCPU that can't be delegated and must be injected. The
    /// interrupt is injected the vCPU is run.
    TimerInterruptEmulation,
}
