/* QEMU mps2-an505, secure aliases: code in SSRAM1 (the CPU boots secure and
   fetches its vector table from 0x10000000), data in SSRAM2. */
MEMORY
{
  FLASH : ORIGIN = 0x10000000, LENGTH = 4M
  RAM   : ORIGIN = 0x38000000, LENGTH = 2M
}
