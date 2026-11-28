/* Minimal layout for size measurement only: code and constants in FLASH,
   initialised data copied to RAM, zeroed data in RAM. */
MEMORY { FLASH : ORIGIN = 0x00000000, LENGTH = 1M  RAM : ORIGIN = 0x20000000, LENGTH = 256K }
ENTRY(_start)
SECTIONS {
  .text : { KEEP(*(.text._start)) *(.text .text.*) } > FLASH
  .rodata : { *(.rodata .rodata.*) } > FLASH
  .data : { *(.data .data.* .sdata .sdata.*) } > RAM AT > FLASH
  .bss (NOLOAD) : { *(.bss .bss.* .sbss .sbss.* COMMON) } > RAM
  /DISCARD/ : { *(.ARM.exidx*) *(.eh_frame*) *(.comment) }
}
