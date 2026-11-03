# Audio sources

The Rust driver and native protocol are new MIT-licensed code in this repository. No Linux or QEMU driver source is compiled or linked into audiod.

The controller registers, ring layouts, reset sequence, PCM format, codec verbs and widget parameters follow the [Intel High Definition Audio Specification 1.0a](https://www.intel.com/content/dam/www/public/us/en/documents/product-specifications/high-definition-audio-specification.pdf), especially sections 3 and 7.

The Cirrus board-specific register facts were checked against the upstream Linux Cirrus codec implementation (GPL-2.0-or-later), [sound/hda/codecs/cirrus/cs420x.c](https://github.com/torvalds/linux/blob/2b414a95b8f7307d42173ba9e580d6d3e2bcbfce/sound/hda/codecs/cirrus/cs420x.c): codec10134208, Mac Pro6,1 subsystem106b7800, GPIO 0 speaker enable, inverted headphone sense, and the CS4208 vendor node24 coefficient33=0001/coefficient34=1c01 processing setup. Audiod expresses these facts in its own small setup routine; it does not import Linux's parser, mixer or fixup framework. Input ADC volume corrections and unrelated Apple models are outside this implementation.

Intel controller1d20 traffic-class and coherent DMA setup (PCI0x44 bits2:0 cleared; PCI0x78 No Snoop bit11 cleared) was checked against upstream Linux [v6.12 hda_intel.c, azx_init_pci](https://github.com/torvalds/linux/blob/v6.12/sound/pci/hda/hda_intel.c). This adjustment is restricted to the supported Mac controller ID.

The QEMU emulator source [intel-hda.c](https://github.com/qemu/qemu/blob/master/hw/audio/intel-hda.c) was consulted to understand its CORB/RIRB response-counter behavior and WAV capture lifecycle. The driver uses standard response status and acknowledgement, with the controller interrupt bit masked while codec replies are polled. QEMU remains an external test tool.

Hardware identity evidence: `results/dross/hardware-20261103/final-memory-fs-netlog.txt` (PCI enumeration), and `results/dross/hardware-20261103/gpu3-netlog.txt` (the earlier Linux CS4208 probe, speaker pin12/headphone pin10). Results are archived outside the branch. Physical route validation is still required.
