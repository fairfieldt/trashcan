/* User programs: static executables at 4 MiB, each segment on pages of its own. */
ENTRY(_start)

PHDRS {
    text PT_LOAD FLAGS(5);
    rodata PT_LOAD FLAGS(4);
    data PT_LOAD FLAGS(6);
}

SECTIONS {
    . = 0x400000;
    .text : ALIGN(4K) { *(.text .text.*) } :text
    .rodata : ALIGN(4K) { *(.rodata .rodata.*) } :rodata
    .data : ALIGN(4K) { *(.data .data.*) *(.got .got.*) } :data
    .bss : ALIGN(4K) { *(.bss .bss.*) *(COMMON) } :data
    /DISCARD/ : { *(.eh_frame*) *(.comment) *(.note*) }
}
