//! How a gateway reaches a part's registers: simulated parts, a serial line,
//! CAN, and Linux I2C/SPI.

pub mod sim;
pub mod uart;

#[cfg(feature = "net")]
#[cfg_attr(docsrs, doc(cfg(feature = "net")))]
pub mod esp_at_twin;
#[cfg(feature = "net")]
#[cfg_attr(docsrs, doc(cfg(feature = "net")))]
pub mod w5500_sim;

#[cfg(feature = "linux")]
#[cfg_attr(docsrs, doc(cfg(feature = "linux")))]
pub mod can;

use crate::gateway::{GatewayError, ToolDevice, Transport};
use crate::regspec::{Device, Engine};

/// Checks a device's signed tool description: the signature against the device's key
/// over the bytes received, and the bytes against the tool list this gateway built from
/// its own spec, so the limits it enforces are exactly the limits the device signed.
pub fn verify_described(dev: &Device, device_key: &str, d: &uart::SignedDescription) -> Result<(), GatewayError> {
    let who = format!("{} ({})", dev.name, dev.title);
    if !crate::auth::verify_tool_description(device_key, dev.name, d.created, &d.tools, &d.signature) {
        return Err(GatewayError::Transport(format!(
            "{who}: the tool description's signature does not verify with the device's key; none of its tools are offered"
        )));
    }
    let mut ours = String::new();
    crate::regspec::engine::write_tools(dev, &mut ours).expect("String");
    if ours.as_bytes() != d.tools.as_slice() {
        return Err(GatewayError::Transport(format!("{who}: the device signed a different tool list from this gateway's spec; none of its tools are offered")));
    }
    Ok(())
}

/// [`open`], and when `device_key` is given, the device's signed tool description is
/// fetched (`D` on the serial protocol) and verified first ([`verify_described`]).
pub fn open_verified(dev: &'static Device<'static>, t: &Transport, device_key: Option<&str>) -> Result<Box<dyn ToolDevice>, GatewayError> {
    let Some(key) = device_key else {
        eprintln!("[webagents-gateway] {}: no device key; its tools are offered under the gateway's own sign-in", dev.name);
        return open(dev, t);
    };
    let err = |e: &dyn std::fmt::Debug| GatewayError::Transport(format!("{} ({}): {e:?}", dev.name, dev.title));
    match t {
        Transport::Tcp { addr, timeout_ms } => {
            let port = std::net::TcpStream::connect(addr.as_str()).map_err(|e| err(&e))?;
            port.set_read_timeout(Some(std::time::Duration::from_millis(timeout_ms.unwrap_or(2000)))).map_err(|e| err(&e))?;
            let _ = port.set_nodelay(true);
            let mut regs = uart::UartRegs { port };
            let d = regs.describe().map_err(|e| err(&e))?;
            verify_described(dev, key, &d)?;
            eprintln!("[webagents-gateway] {}: tool description signed by the device's key, verified", dev.name);
            Ok(Box::new(Engine::new(dev, regs, vec![0u64; dev.field_count])))
        }
        #[cfg(feature = "linux")]
        Transport::Uart { path, baud, timeout_ms } => {
            let port = serialport::new(path, baud.unwrap_or(115_200))
                .timeout(std::time::Duration::from_millis(timeout_ms.unwrap_or(200)))
                .open_native()
                .map_err(|e| err(&e))?;
            let mut regs = uart::UartRegs { port };
            let d = regs.describe().map_err(|e| err(&e))?;
            verify_described(dev, key, &d)?;
            eprintln!("[webagents-gateway] {}: tool description signed by the device's key, verified", dev.name);
            Ok(Box::new(Engine::new(dev, regs, vec![0u64; dev.field_count])))
        }
        _ => Err(GatewayError::Config(format!("{}: a device key needs a serial transport (uart or tcp) to fetch the signed description", dev.name))),
    }
}

/// Opens the transport for one device and wraps it in its engine.
pub fn open(dev: &'static Device<'static>, t: &Transport) -> Result<Box<dyn ToolDevice>, GatewayError> {
    let state = vec![0u64; dev.field_count];
    match t {
        Transport::Sim => Ok(Box::new(Engine::new(dev, sim::SimBus(sim::RegModel::at_reset(dev)), state))),
        Transport::Tcp { addr, timeout_ms } => {
            let err = |e: std::io::Error| GatewayError::Transport(format!("{} ({}): {addr}: {e}", dev.name, dev.title));
            let port = std::net::TcpStream::connect(addr.as_str()).map_err(err)?;
            port.set_read_timeout(Some(std::time::Duration::from_millis(timeout_ms.unwrap_or(2000)))).map_err(err)?;
            let _ = port.set_nodelay(true);
            Ok(Box::new(Engine::new(dev, uart::UartRegs { port }, state)))
        }
        #[cfg(feature = "linux")]
        other => linux::open(dev, other, state),
        #[cfg(not(feature = "linux"))]
        _ => Err(GatewayError::Config("this build has no Linux transports (enable the `linux` feature)".into())),
    }
}

#[cfg(feature = "linux")]
mod linux {
    use std::time::Duration;

    use super::*;
    use crate::bus::{I2cRegs, SpiRegs};
    use crate::regspec::Bus;

    pub fn open(dev: &'static Device<'static>, t: &Transport, state: Vec<u64>) -> Result<Box<dyn ToolDevice>, GatewayError> {
        let err = |what: &str, e: &dyn std::fmt::Display| GatewayError::Transport(format!("{} ({}): {what}: {e}", dev.name, dev.title));
        match t {
            Transport::Sim | Transport::Tcp { .. } => unreachable!("handled by the caller"),
            Transport::Uart { path, baud, timeout_ms } => {
                let port = serialport::new(path, baud.unwrap_or(115_200))
                    .timeout(Duration::from_millis(timeout_ms.unwrap_or(200)))
                    .open_native()
                    .map_err(|e| err(path, &e))?;
                Ok(Box::new(Engine::new(dev, super::uart::UartRegs { port }, state)))
            }
            Transport::Can { iface, node } => {
                let bus = super::can::CanRegs::open(iface, *node).map_err(|e| err(iface, &e))?;
                Ok(Box::new(Engine::new(dev, bus, state)))
            }
            Transport::I2c { bus, address } => {
                let Bus::I2c { address: spec_addr, addr_bytes, data_bytes, big_endian } = dev.bus else {
                    return Err(err(bus, &"the spec does not describe an I2C part"));
                };
                let i2c = linux_embedded_hal::I2cdev::new(bus).map_err(|e| err(bus, &e))?;
                Ok(Box::new(Engine::new(dev, I2cRegs { i2c, address: address.unwrap_or(spec_addr), addr_bytes, data_bytes, big_endian }, state)))
            }
            Transport::Spi { path } => {
                let Bus::Spi { addr_bytes, data_bytes, big_endian } = dev.bus else {
                    return Err(err(path, &"the spec does not describe a SPI part"));
                };
                let spi = linux_embedded_hal::SpidevDevice::open(path).map_err(|e| err(path, &e))?;
                Ok(Box::new(Engine::new(dev, SpiRegs { spi, addr_bytes, data_bytes, big_endian, read_bit: 0x80 }, state)))
            }
        }
    }
}
