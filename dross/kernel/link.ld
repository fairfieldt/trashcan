ENTRY(kernel_entry)

KERNEL_BASE = 0xffffffff80000000;

PHDRS {
    text PT_LOAD FLAGS(5);
    rodata PT_LOAD FLAGS(4);
    data PT_LOAD FLAGS(6);
}

SECTIONS {
    . = KERNEL_BASE;
    .text : ALIGN(4K) { *(.text.entry) *(.text .text.*) } :text
    .rodata : ALIGN(4K) { *(.rodata .rodata.*) } :rodata
    .data : ALIGN(4K) { *(.data .data.*) *(.got .got.*) } :data
    .bss : ALIGN(4K) { *(.bss .bss.*) *(COMMON) } :data
    /DISCARD/ : { *(.eh_frame*) *(.comment) *(.note*) }
}
