	.file	"stage2.cc"
	.section	.text._ZNKSt5ctypeIcE8do_widenEc,"axG",@progbits,_ZNKSt5ctypeIcE8do_widenEc,comdat
	.align 2
	.p2align 4,,15
	.weak	_ZNKSt5ctypeIcE8do_widenEc
	.type	_ZNKSt5ctypeIcE8do_widenEc, @function
_ZNKSt5ctypeIcE8do_widenEc:
.LFB1820:
	.cfi_startproc
	movl	%esi, %eax
	ret
	.cfi_endproc
.LFE1820:
	.size	_ZNKSt5ctypeIcE8do_widenEc, .-_ZNKSt5ctypeIcE8do_widenEc
	.text
	.p2align 4,,15
	.globl	_Z18stage_2_block_sizedd
	.type	_Z18stage_2_block_sizedd, @function
_Z18stage_2_block_sizedd:
.LFB2881:
	.cfi_startproc
	movapd	%xmm0, %xmm2
	sqrtsd	%xmm1, %xmm1
	sqrtsd	%xmm1, %xmm1
	mulsd	.LC1(%rip), %xmm0
	movl	$5120, %edx
	divsd	%xmm1, %xmm2
	minsd	%xmm2, %xmm0
	cvttsd2siq	%xmm0, %rax
	cmpl	$5120, %eax
	cmova	%edx, %eax
	ret
	.cfi_endproc
.LFE2881:
	.size	_Z18stage_2_block_sizedd, .-_Z18stage_2_block_sizedd
	.p2align 4,,15
	.globl	_Z13estimate_sizeP12__mpz_structP13__mpfr_structj
	.type	_Z13estimate_sizeP12__mpz_structP13__mpfr_structj, @function
_Z13estimate_sizeP12__mpz_structP13__mpfr_structj:
.LFB2885:
	.cfi_startproc
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	movq	%rsi, %r13
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	movq	%rdi, %r12
	movl	%edx, %ebx
	subq	$40, %rsp
	.cfi_def_cfa_offset 80
	leaq	16(%rsp), %rbp
	movq	%rbp, %rdi
	call	__gmpz_init@PLT
	xorl	%esi, %esi
	movq	%r13, %rdi
	call	mpfr_get_d@PLT
	movq	%r12, %rsi
	movq	%rbp, %rdi
	movsd	%xmm0, 8(%rsp)
	xorl	%r12d, %r12d
	call	__gmpz_set@PLT
	testl	%ebx, %ebx
	je	.L3
	movsd	8(%rsp), %xmm0
	movsd	.LC0(%rip), %xmm1
	sqrtsd	%xmm0, %xmm0
	sqrtsd	%xmm0, %xmm0
	divsd	%xmm0, %xmm1
	movsd	%xmm1, 8(%rsp)
	.p2align 4,,10
	.p2align 3
.L7:
	movq	%rbp, %rdi
	call	__gmpz_get_d@PLT
	cmpl	$1, %ebx
	jbe	.L12
	movsd	8(%rsp), %xmm1
	cmpl	$5120, %ebx
	movl	$5120, %eax
	cmovbe	%ebx, %eax
	movq	%rbp, %rsi
	movq	%rbp, %rdi
	mulsd	%xmm0, %xmm1
	mulsd	.LC1(%rip), %xmm0
	minsd	%xmm1, %xmm0
	cvttsd2siq	%xmm0, %rdx
	cmpl	%eax, %edx
	cmova	%eax, %edx
	addl	$1, %r12d
	subl	%edx, %ebx
	movl	%edx, %edx
	call	__gmpz_add_ui@PLT
	testl	%ebx, %ebx
	jne	.L7
.L3:
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	movl	%r12d, %eax
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L12:
	.cfi_restore_state
	movq	%rbp, %rsi
	movq	%rbp, %rdi
	movl	$1, %edx
	call	__gmpz_add_ui@PLT
	addl	$1, %r12d
	addq	$40, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	movl	%r12d, %eax
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE2885:
	.size	_Z13estimate_sizeP12__mpz_structP13__mpfr_structj, .-_Z13estimate_sizeP12__mpz_structP13__mpfr_structj
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC2:
	.string	"stage2.cc"
.LC3:
	.string	"cudaFree device"
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC4:
	.string	"Fatal error: %s (%s at %s:%d)\n"
	.section	.rodata.str1.1
.LC5:
	.string	"*** FAILED - ABORTING\n"
.LC6:
	.string	"cudaFree host"
.LC7:
	.string	"cudaMalloc device"
.LC8:
	.string	"cudaMalloc host"
	.text
	.p2align 4,,15
	.globl	_Z10reallocatePP20precomputation_tablePP7double2S1_S4_PPSt7complexIdEi
	.type	_Z10reallocatePP20precomputation_tablePP7double2S1_S4_PPSt7complexIdEi, @function
_Z10reallocatePP20precomputation_tablePP7double2S1_S4_PPSt7complexIdEi:
.LFB2886:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	movq	%rdi, %r15
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	movq	%rdx, %r14
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rsi, %rbp
	movq	%rcx, %r13
	movq	%r8, %r12
	movslq	%r9d, %rbx
	subq	$24, %rsp
	.cfi_def_cfa_offset 80
	movq	(%rdi), %rdi
	call	cudaFree@PLT
	movq	0(%rbp), %rdi
	call	cudaFree@PLT
	call	cudaGetLastError@PLT
	testl	%eax, %eax
	jne	.L20
	movq	(%r14), %rdi
	call	cudaFreeHost@PLT
	movq	0(%r13), %rdi
	call	cudaFreeHost@PLT
	movq	(%r12), %rdi
	call	cudaFreeHost@PLT
	call	cudaGetLastError@PLT
	testl	%eax, %eax
	jne	.L21
	leaq	(%rbx,%rbx,8), %rdx
	movq	%r15, %rdi
	salq	$4, %rbx
	salq	$4, %rdx
	movq	%rdx, %rsi
	movq	%rdx, 8(%rsp)
	call	cudaMalloc@PLT
	movq	%rbx, %rsi
	movq	%rbp, %rdi
	call	cudaMalloc@PLT
	call	cudaGetLastError@PLT
	testl	%eax, %eax
	movq	8(%rsp), %rdx
	jne	.L22
	movq	%rdx, %rsi
	movq	%r14, %rdi
	call	cudaMallocHost@PLT
	movq	%rbx, %rsi
	movq	%r13, %rdi
	call	cudaMallocHost@PLT
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	cudaMallocHost@PLT
	call	cudaGetLastError@PLT
	testl	%eax, %eax
	jne	.L23
	addq	$24, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.L20:
	.cfi_restore_state
	movl	%eax, %edi
	call	cudaGetErrorString@PLT
	leaq	.LC2(%rip), %r8
	leaq	.LC3(%rip), %rdx
	movl	$296, %r9d
	movq	%rax, %rcx
.L19:
	movq	stderr(%rip), %rdi
	leaq	.LC4(%rip), %rsi
	xorl	%eax, %eax
	call	fprintf@PLT
	movq	stderr(%rip), %rcx
	leaq	.LC5(%rip), %rdi
	movl	$22, %edx
	movl	$1, %esi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L23:
	movl	%eax, %edi
	call	cudaGetErrorString@PLT
	movl	$310, %r9d
	leaq	.LC2(%rip), %r8
	movq	%rax, %rcx
	leaq	.LC8(%rip), %rdx
	jmp	.L19
.L22:
	movl	%eax, %edi
	call	cudaGetErrorString@PLT
	movl	$305, %r9d
	leaq	.LC2(%rip), %r8
	movq	%rax, %rcx
	leaq	.LC7(%rip), %rdx
	jmp	.L19
.L21:
	movl	%eax, %edi
	call	cudaGetErrorString@PLT
	movl	$301, %r9d
	leaq	.LC2(%rip), %r8
	movq	%rax, %rcx
	leaq	.LC6(%rip), %rdx
	jmp	.L19
	.cfi_endproc
.LFE2886:
	.size	_Z10reallocatePP20precomputation_tablePP7double2S1_S4_PPSt7complexIdEi, .-_Z10reallocatePP20precomputation_tablePP7double2S1_S4_PPSt7complexIdEi
	.section	.rodata.str1.1
.LC9:
	.string	"texture"
	.text
	.p2align 4,,15
	.globl	_Z8allocatePP20precomputation_tablePP7double2S1_S4_PPSt7complexIdEi
	.type	_Z8allocatePP20precomputation_tablePP7double2S1_S4_PPSt7complexIdEi, @function
_Z8allocatePP20precomputation_tablePP7double2S1_S4_PPSt7complexIdEi:
.LFB2887:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	movq	%rsi, %r14
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	movq	%rdx, %r13
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rcx, %r12
	movq	%r8, %rbp
	movslq	%r9d, %rbx
	subq	$24, %rsp
	.cfi_def_cfa_offset 80
	movq	%rdi, 8(%rsp)
	call	_Z15allocateTexturev@PLT
	call	cudaGetLastError@PLT
	testl	%eax, %eax
	movq	8(%rsp), %rdi
	jne	.L30
	leaq	(%rbx,%rbx,8), %r15
	salq	$4, %rbx
	salq	$4, %r15
	movq	%r15, %rsi
	call	cudaMalloc@PLT
	movq	%rbx, %rsi
	movq	%r14, %rdi
	call	cudaMalloc@PLT
	call	cudaGetLastError@PLT
	testl	%eax, %eax
	jne	.L31
	movq	%r15, %rsi
	movq	%r13, %rdi
	call	cudaMallocHost@PLT
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	cudaMallocHost@PLT
	movq	%rbx, %rsi
	movq	%rbp, %rdi
	call	cudaMallocHost@PLT
	call	cudaGetLastError@PLT
	testl	%eax, %eax
	jne	.L32
	addq	$24, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.L30:
	.cfi_restore_state
	movl	%eax, %edi
	call	cudaGetErrorString@PLT
	leaq	.LC2(%rip), %r8
	leaq	.LC9(%rip), %rdx
	movl	$325, %r9d
	movq	%rax, %rcx
.L29:
	movq	stderr(%rip), %rdi
	leaq	.LC4(%rip), %rsi
	xorl	%eax, %eax
	call	fprintf@PLT
	movq	stderr(%rip), %rcx
	leaq	.LC5(%rip), %rdi
	movl	$22, %edx
	movl	$1, %esi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L32:
	movl	%eax, %edi
	call	cudaGetErrorString@PLT
	movl	$335, %r9d
	leaq	.LC2(%rip), %r8
	movq	%rax, %rcx
	leaq	.LC8(%rip), %rdx
	jmp	.L29
.L31:
	movl	%eax, %edi
	call	cudaGetErrorString@PLT
	movl	$330, %r9d
	leaq	.LC2(%rip), %r8
	movq	%rax, %rcx
	leaq	.LC7(%rip), %rdx
	jmp	.L29
	.cfi_endproc
.LFE2887:
	.size	_Z8allocatePP20precomputation_tablePP7double2S1_S4_PPSt7complexIdEi, .-_Z8allocatePP20precomputation_tablePP7double2S1_S4_PPSt7complexIdEi
	.section	.rodata.str1.8
	.align 8
.LC11:
	.string	"Error: in stage 2, computed a block size that is too small "
	.align 8
.LC12:
	.string	"Refusing to continue, and returning NAN, without setting K."
	.globl	__divtf3
	.globl	__multf3
	.globl	__floatsitf
	.globl	__trunctfdf2
	.globl	__addtf3
	.align 8
.LC16:
	.string	"Something is seriously messed up\n"
	.section	.rodata.str1.1
.LC17:
	.string	"memcpyAsync to device\n"
.LC18:
	.string	"cudaKernel launch"
.LC19:
	.string	"memcpyAsync from device\n"
.LC20:
	.string	"cudaStream synchronize\n"
	.text
	.p2align 4,,15
	.globl	_Z17zeta_block_stage2P12__mpz_structjP13__mpfr_structdiPSt7complexIdEPP20precomputation_tablePP7double2S8_SB_PS5_iP11CUstream_stiP15pthread_mutex_ti
	.type	_Z17zeta_block_stage2P12__mpz_structjP13__mpfr_structdiPSt7complexIdEPP20precomputation_tablePP7double2S8_SB_PS5_iP11CUstream_stiP15pthread_mutex_ti, @function
_Z17zeta_block_stage2P12__mpz_structjP13__mpfr_structdiPSt7complexIdEPP20precomputation_tablePP7double2S8_SB_PS5_iP11CUstream_stiP15pthread_mutex_ti:
.LFB2888:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$632, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	testl	%esi, %esi
	movq	%rdi, -672(%rbp)
	movl	%esi, -620(%rbp)
	movq	%rdx, -400(%rbp)
	movsd	%xmm0, -608(%rbp)
	movl	%ecx, -556(%rbp)
	movq	%r8, -600(%rbp)
	movq	%r9, -664(%rbp)
	movl	$0, -616(%rbp)
	jne	.L143
.L33:
	movl	-616(%rbp), %eax
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L143:
	.cfi_restore_state
	leaq	-304(%rbp), %r12
	movq	%r12, %rdi
	call	__gmpz_init@PLT
	movq	-672(%rbp), %rsi
	movq	%r12, %rdi
	call	__gmpz_set@PLT
	movl	-620(%rbp), %edx
	movq	-400(%rbp), %rsi
	movq	%r12, %rdi
	call	_Z13estimate_sizeP12__mpz_structP13__mpfr_structj
	movl	80(%rbp), %r10d
	movl	%eax, -616(%rbp)
	testl	%r10d, %r10d
	je	.L144
	cmpl	80(%rbp), %eax
	jle	.L39
	movl	80(%rbp), %r9d
	testl	%r9d, %r9d
	jle	.L38
.L162:
	movl	-616(%rbp), %r9d
	movq	40(%rbp), %r8
	movq	32(%rbp), %rcx
	movq	24(%rbp), %rdx
	movq	16(%rbp), %rsi
	movq	-664(%rbp), %rdi
	call	_Z10reallocatePP20precomputation_tablePP7double2S1_S4_PPSt7complexIdEi
.L38:
	movl	-620(%rbp), %eax
	movq	$0, -552(%rbp)
	movl	$0, -612(%rbp)
	movq	%r12, -504(%rbp)
	movl	%eax, -444(%rbp)
	.p2align 4,,10
	.p2align 3
.L102:
	movq	24(%rbp), %rax
	movq	-552(%rbp), %rbx
	movl	-444(%rbp), %r15d
	movq	%rsp, -528(%rbp)
	addq	(%rax), %rbx
	cmpl	$1, %r15d
	movl	$0, (%rbx)
	movl	$0, 4(%rbx)
	movl	$0, 8(%rbx)
	jbe	.L121
	movq	-504(%rbp), %rdi
	call	__gmpz_get_d@PLT
	movq	-400(%rbp), %rdi
	xorl	%esi, %esi
	movsd	%xmm0, -512(%rbp)
	call	mpfr_get_d@PLT
	movsd	.LC0(%rip), %xmm7
	movl	$5120, %eax
	movsd	%xmm0, -440(%rbp)
	sqrtsd	%xmm0, %xmm0
	sqrtsd	%xmm0, %xmm0
	divsd	%xmm0, %xmm7
	movsd	-512(%rbp), %xmm6
	movsd	.LC1(%rip), %xmm1
	movapd	%xmm7, %xmm0
	mulsd	%xmm6, %xmm1
	mulsd	%xmm6, %xmm0
	minsd	%xmm1, %xmm0
	cvttsd2siq	%xmm0, %rdx
	cmpl	$5120, %edx
	cmovbe	%edx, %eax
	cmpl	%r15d, %eax
	movl	%eax, -488(%rbp)
	ja	.L41
	cmpl	$1, %eax
	movl	%eax, 8(%rbx)
	jbe	.L145
	subl	%eax, -444(%rbp)
.L118:
	pxor	%xmm0, %xmm0
	cvtsi2sd	%eax, %xmm0
	movq	-440(%rbp), %rax
	sarq	$52, %rax
	subq	$1023, %rax
	testq	%rax, %rax
	movl	%eax, %r13d
	divsd	-512(%rbp), %xmm0
	movq	%xmm0, %r12
	je	.L146
.L49:
	movq	%r12, %rax
	sarq	$52, %rax
	subq	$1023, %rax
	testq	%rax, %rax
	movl	%eax, %ecx
	je	.L147
.L50:
	movl	$-58, %esi
	movq	-504(%rbp), %rdi
	movl	%esi, %eax
	subl	%r13d, %eax
	cltd
	idivl	%ecx
	movl	%eax, -328(%rbp)
	movl	%esi, %eax
	movl	$2, %esi
	cltd
	idivl	%ecx
	movl	%eax, -560(%rbp)
	movl	%eax, 4(%rbx)
	movl	-328(%rbp), %eax
	movl	%eax, (%rbx)
	call	__gmpz_sizeinbase@PLT
	movq	-400(%rbp), %rcx
	movl	%eax, -448(%rbp)
	movq	16(%rcx), %rcx
	movq	%rcx, -352(%rbp)
	movl	-352(%rbp), %ecx
	leal	53(%rcx), %edx
	subl	%eax, %edx
	cmpl	$105, %edx
	movl	%edx, -484(%rbp)
	movslq	%edx, %r12
	jg	.L148
.L52:
	movl	%edx, %eax
	movq	-504(%rbp), %rsi
	movabsq	$-9223372036854775806, %r14
	addl	$62, %eax
	subl	$1, %edx
	movq	$0, -104(%rbp)
	cmovns	%edx, %eax
	xorl	%edx, %edx
	movl	$1, -104(%rbp)
	sarl	$6, %eax
	movq	%r14, -96(%rbp)
	movq	%r12, -112(%rbp)
	addl	$1, %eax
	cltq
	leaq	22(,%rax,8), %rax
	andq	$-16, %rax
	subq	%rax, %rsp
	leaq	-112(%rbp), %rax
	movq	%rsp, -88(%rbp)
	movq	%rax, %rdi
	movq	%rax, -520(%rbp)
	call	mpfr_set_z@PLT
	movq	-96(%rbp), %rcx
	cmpq	%r14, %rcx
	je	.L149
	movabsq	$-9223372036854775805, %rax
	cmpq	%rax, %rcx
	je	.L150
	movabsq	$-9223372036854775807, %rax
	cmpq	%rax, %rcx
	je	.L151
	xorl	%esi, %esi
	cmpl	$-1, -104(%rbp)
	movabsq	$9223372036854775807, %rdx
	sete	%sil
	movq	%rsi, %rax
	movq	-472(%rbp), %rsi
	salq	$63, %rax
	andq	%rdx, %rsi
	orq	%rax, %rsi
	movq	-88(%rbp), %rax
	cmpq	$-16381, %rcx
	movq	%rsi, -472(%rbp)
	movq	8(%rax), %rdx
	movq	(%rax), %rax
	jge	.L57
	movq	$-16381, %rdi
	subq	%rcx, %rdi
	xorl	%ecx, %ecx
	.p2align 4,,10
	.p2align 3
.L58:
	movq	%rdx, %rsi
	shrq	%rax
	addq	$1, %rcx
	salq	$63, %rsi
	shrq	%rdx
	orq	%rsi, %rax
	cmpq	%rcx, %rdi
	jne	.L58
	movq	-472(%rbp), %rsi
	movabsq	$-9223090561878065153, %rcx
	andq	%rcx, %rsi
.L59:
	leaq	(%rdx,%rdx), %rcx
	movabsq	$-281474976710656, %rdi
	salq	$49, %rdx
	andq	%rdi, %rsi
	shrq	$15, %rax
	shrq	$16, %rcx
	orq	%rax, %rdx
	orq	%rcx, %rsi
	movq	%rdx, -480(%rbp)
	movq	%rsi, -472(%rbp)
	movdqa	-480(%rbp), %xmm1
.L54:
	cmpl	$105, -484(%rbp)
	jg	.L152
	movdqa	.LC13(%rip), %xmm0
	call	__divtf3@PLT
	movsd	.LC0(%rip), %xmm6
	cmpl	$52, -484(%rbp)
	movaps	%xmm0, -576(%rbp)
	divsd	-512(%rbp), %xmm6
	movsd	%xmm6, -456(%rbp)
	jg	.L153
	movq	$0, -592(%rbp)
	movq	$0, -584(%rbp)
	movq	$0, -544(%rbp)
	movq	$0, -536(%rbp)
	movq	$0, -352(%rbp)
	movq	$0, -344(%rbp)
.L69:
	movl	-328(%rbp), %r8d
	testl	%r8d, %r8d
	jle	.L98
	leaq	-272(%rbp), %rax
	movdqa	-544(%rbp), %xmm6
	movl	-484(%rbp), %r13d
	movl	$1, %r14d
	movl	$1, %r15d
	movq	%rax, -336(%rbp)
	leaq	-208(%rbp), %rax
	leaq	-240(%rbp), %r12
	movsd	-456(%rbp), %xmm2
	movq	%rax, -392(%rbp)
	leaq	-176(%rbp), %rax
	movaps	%xmm6, -432(%rbp)
	movq	%rax, -464(%rbp)
	jmp	.L99
	.p2align 4,,10
	.p2align 3
.L81:
	cmpl	$52, %r13d
	jg	.L154
	pxor	%xmm0, %xmm0
	movsd	-440(%rbp), %xmm1
	cvtsi2sd	%r15d, %xmm0
	mulsd	%xmm2, %xmm1
	mulsd	-456(%rbp), %xmm2
	mulsd	%xmm1, %xmm0
	pxor	%xmm1, %xmm1
	cvtsi2sd	%r14d, %xmm1
	divsd	%xmm1, %xmm0
	movsd	%xmm0, 16(%rbx,%r14,8)
.L82:
	negl	%r15d
	addq	$1, %r14
	cmpl	%r14d, -328(%rbp)
	jl	.L98
.L99:
	cmpl	$105, %r13d
	jle	.L81
	movq	-336(%rbp), %rdi
	movslq	%r13d, %rax
	movsd	%xmm2, -496(%rbp)
	movq	%rax, %rsi
	movq	%rax, -416(%rbp)
	call	mpfr_set_prec@PLT
	movq	-416(%rbp), %rsi
	xorl	%edx, %edx
	movq	%r12, %rdi
	call	mpfr_prec_round@PLT
	movq	-416(%rbp), %rsi
	movq	-392(%rbp), %rdi
	xorl	%edx, %edx
	call	mpfr_prec_round@PLT
	movq	-400(%rbp), %rsi
	movq	-336(%rbp), %rdi
	xorl	%ecx, %ecx
	movq	%r12, %rdx
	call	mpfr_mul@PLT
	movq	-392(%rbp), %rdx
	movq	-336(%rbp), %rsi
	xorl	%ecx, %ecx
	movq	-464(%rbp), %rdi
	call	mpfr_fmod@PLT
	movq	-464(%rbp), %rdi
	xorl	%esi, %esi
	call	mpfr_get_d@PLT
	pxor	%xmm1, %xmm1
	movq	-520(%rbp), %rdx
	xorl	%ecx, %ecx
	movq	%r12, %rsi
	movq	%r12, %rdi
	cvtsi2sd	%r15d, %xmm1
	mulsd	%xmm1, %xmm0
	pxor	%xmm1, %xmm1
	cvtsi2sd	%r14d, %xmm1
	divsd	%xmm1, %xmm0
	movsd	%xmm0, 16(%rbx,%r14,8)
	call	mpfr_mul@PLT
	movq	-392(%rbp), %rdi
	leaq	-144(%rbp), %rdx
	xorl	%ecx, %ecx
	movq	%rdi, %rsi
	call	mpfr_add@PLT
	subl	-448(%rbp), %r13d
	movsd	-496(%rbp), %xmm2
	cmpl	$105, %r13d
	jg	.L82
	movq	-224(%rbp), %rcx
	movabsq	$-9223372036854775806, %rax
	cmpq	%rax, %rcx
	je	.L155
	movabsq	$-9223372036854775805, %rax
	cmpq	%rax, %rcx
	je	.L156
	movabsq	$-9223372036854775807, %rax
	cmpq	%rax, %rcx
	je	.L157
	xorl	%esi, %esi
	cmpl	$-1, -232(%rbp)
	movabsq	$9223372036854775807, %rdx
	sete	%sil
	movq	%rsi, %rax
	movq	-360(%rbp), %rsi
	salq	$63, %rax
	andq	%rdx, %rsi
	orq	%rax, %rsi
	movq	-216(%rbp), %rax
	cmpq	$-16381, %rcx
	movq	%rsi, -360(%rbp)
	movq	8(%rax), %rdx
	movq	(%rax), %rax
	jge	.L87
	movq	$-16381, %rdi
	subq	%rcx, %rdi
	xorl	%ecx, %ecx
	.p2align 4,,10
	.p2align 3
.L88:
	movq	%rdx, %rsi
	shrq	%rax
	addq	$1, %rcx
	salq	$63, %rsi
	shrq	%rdx
	orq	%rsi, %rax
	cmpq	%rcx, %rdi
	jne	.L88
	movq	-360(%rbp), %rsi
	movabsq	$-9223090561878065153, %rcx
	andq	%rcx, %rsi
.L89:
	leaq	(%rdx,%rdx), %rcx
	movabsq	$-281474976710656, %rdi
	salq	$49, %rdx
	andq	%rdi, %rsi
	shrq	$15, %rax
	shrq	$16, %rcx
	orq	%rax, %rdx
	orq	%rcx, %rsi
	movq	%rdx, -368(%rbp)
	movq	%rsi, -360(%rbp)
	movdqa	-368(%rbp), %xmm6
	movaps	%xmm6, -352(%rbp)
.L84:
	movq	-192(%rbp), %rcx
	movabsq	$-9223372036854775806, %rax
	cmpq	%rax, %rcx
	je	.L158
	movabsq	$-9223372036854775805, %rax
	cmpq	%rax, %rcx
	je	.L159
	movabsq	$-9223372036854775807, %rax
	cmpq	%rax, %rcx
	je	.L160
	xorl	%esi, %esi
	cmpl	$-1, -200(%rbp)
	movabsq	$9223372036854775807, %rdx
	sete	%sil
	movq	%rsi, %rax
	movq	-376(%rbp), %rsi
	salq	$63, %rax
	andq	%rdx, %rsi
	orq	%rax, %rsi
	movq	-184(%rbp), %rax
	cmpq	$-16381, %rcx
	movq	%rsi, -376(%rbp)
	movq	8(%rax), %rdx
	movq	(%rax), %rax
	jge	.L94
	movq	$-16381, %rdi
	subq	%rcx, %rdi
	xorl	%ecx, %ecx
	.p2align 4,,10
	.p2align 3
.L95:
	movq	%rdx, %rsi
	shrq	%rax
	addq	$1, %rcx
	salq	$63, %rsi
	shrq	%rdx
	orq	%rsi, %rax
	cmpq	%rcx, %rdi
	jne	.L95
	movq	-376(%rbp), %rsi
	movabsq	$-9223090561878065153, %rcx
	andq	%rcx, %rsi
.L96:
	leaq	(%rdx,%rdx), %rcx
	movabsq	$-281474976710656, %rdi
	salq	$49, %rdx
	andq	%rdi, %rsi
	shrq	$15, %rax
	shrq	$16, %rcx
	orq	%rax, %rdx
	orq	%rcx, %rsi
	movq	%rdx, -384(%rbp)
	movq	%rsi, -376(%rbp)
	movdqa	-384(%rbp), %xmm7
	movaps	%xmm7, -432(%rbp)
.L91:
	cmpl	$52, %r13d
	jg	.L82
	xorl	%esi, %esi
	movq	%r12, %rdi
	negl	%r15d
	call	mpfr_get_d@PLT
	addq	$1, %r14
	cmpl	%r14d, -328(%rbp)
	movapd	%xmm0, %xmm2
	jge	.L99
	.p2align 4,,10
	.p2align 3
.L98:
	movl	-560(%rbp), %edi
	testl	%edi, %edi
	jle	.L100
	movsd	.LC14(%rip), %xmm3
	movl	-560(%rbp), %edx
	movl	$1, %eax
	movsd	.LC0(%rip), %xmm1
	divsd	-512(%rbp), %xmm3
	movsd	.LC15(%rip), %xmm4
	.p2align 4,,10
	.p2align 3
.L101:
	pxor	%xmm2, %xmm2
	mulsd	%xmm3, %xmm1
	cvtsi2sd	%eax, %xmm2
	movapd	%xmm1, %xmm0
	mulsd	%xmm4, %xmm0
	divsd	%xmm2, %xmm0
	movsd	%xmm0, 80(%rbx,%rax,8)
	addq	$1, %rax
	cmpl	%eax, %edx
	jge	.L101
.L100:
	cmpl	$105, -484(%rbp)
	jg	.L161
.L80:
	movq	-504(%rbp), %rdi
	movl	-488(%rbp), %edx
	movq	-528(%rbp), %rsp
	addl	$1, -612(%rbp)
	movq	%rdi, %rsi
	call	__gmpz_add_ui@PLT
	movl	-444(%rbp), %esi
	addq	$144, -552(%rbp)
	testl	%esi, %esi
	jne	.L102
	movq	-504(%rbp), %r12
	jmp	.L119
	.p2align 4,,10
	.p2align 3
.L144:
	movq	40(%rbp), %r8
	movq	32(%rbp), %rcx
	movl	%eax, %r9d
	movq	24(%rbp), %rdx
	movq	16(%rbp), %rsi
	movl	%eax, %ebx
	movq	-664(%rbp), %rdi
	call	_Z8allocatePP20precomputation_tablePP7double2S1_S4_PPSt7complexIdEi
	movq	56(%rbp), %rdi
	call	cudaStreamSynchronize@PLT
	testl	%ebx, %ebx
	jg	.L38
.L39:
	movl	-616(%rbp), %eax
	addl	%eax, %eax
	cmpl	%eax, 80(%rbp)
	jle	.L38
	movl	80(%rbp), %r9d
	testl	%r9d, %r9d
	jg	.L162
	jmp	.L38
	.p2align 4,,10
	.p2align 3
.L154:
	movdqa	-592(%rbp), %xmm1
	movdqa	-352(%rbp), %xmm0
	movsd	%xmm2, -496(%rbp)
	call	__multf3@PLT
	movdqa	-432(%rbp), %xmm1
	call	fmodq@PLT
	movl	%r15d, %edi
	movaps	%xmm0, -416(%rbp)
	call	__floatsitf@PLT
	movdqa	-416(%rbp), %xmm1
	call	__multf3@PLT
	movl	%r14d, %edi
	movaps	%xmm0, -416(%rbp)
	call	__floatsitf@PLT
	movdqa	%xmm0, %xmm1
	movdqa	-416(%rbp), %xmm0
	call	__divtf3@PLT
	call	__trunctfdf2@PLT
	movdqa	-576(%rbp), %xmm1
	movsd	%xmm0, 16(%rbx,%r14,8)
	movdqa	-352(%rbp), %xmm0
	call	__multf3@PLT
	movdqa	-544(%rbp), %xmm1
	movaps	%xmm0, -352(%rbp)
	movdqa	-432(%rbp), %xmm0
	call	__addtf3@PLT
	subl	-448(%rbp), %r13d
	movsd	-496(%rbp), %xmm2
	movaps	%xmm0, -432(%rbp)
	cmpl	$52, %r13d
	jg	.L82
	movdqa	-352(%rbp), %xmm0
	call	__trunctfdf2@PLT
	movapd	%xmm0, %xmm2
	jmp	.L82
	.p2align 4,,10
	.p2align 3
.L121:
	movq	-504(%rbp), %r12
	movl	$1, -444(%rbp)
.L40:
	movl	-444(%rbp), %edx
	movq	-528(%rbp), %rsp
	movq	%r12, %rsi
	movq	%r12, %rdi
	addl	$1, -612(%rbp)
	call	__gmpz_add_ui@PLT
.L119:
	movl	-612(%rbp), %ebx
	cmpl	%ebx, -616(%rbp)
	je	.L103
	leaq	.LC16(%rip), %rsi
	leaq	_ZSt4cout(%rip), %rdi
	movl	$33, %edx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.L103:
	movq	24(%rbp), %rax
	movslq	-616(%rbp), %rbx
	movl	$1, %ecx
	movq	56(%rbp), %r8
	movq	(%rax), %rsi
	movq	-664(%rbp), %rax
	leaq	(%rbx,%rbx,8), %rdx
	movq	(%rax), %rdi
	salq	$4, %rdx
	call	cudaMemcpyAsync@PLT
	call	cudaGetLastError@PLT
	testl	%eax, %eax
	jne	.L163
	movq	16(%rbp), %rax
	movl	64(%rbp), %r8d
	movq	56(%rbp), %rcx
	movl	-616(%rbp), %edx
	movq	(%rax), %rsi
	movq	-664(%rbp), %rax
	movq	(%rax), %rdi
	call	_Z10cudaKernelP20precomputation_tableP7double2iP11CUstream_sti@PLT
	call	cudaGetLastError@PLT
	testl	%eax, %eax
	jne	.L164
	movq	16(%rbp), %rax
	movq	56(%rbp), %r8
	salq	$4, %rbx
	movq	%rbx, %rdx
	movl	$2, %ecx
	movq	(%rax), %rsi
	movq	32(%rbp), %rax
	movq	(%rax), %rdi
	call	cudaMemcpyAsync@PLT
	call	cudaGetLastError@PLT
	testl	%eax, %eax
	jne	.L165
	movq	56(%rbp), %rdi
	call	cudaStreamSynchronize@PLT
	call	cudaGetLastError@PLT
	testl	%eax, %eax
	jne	.L107
	movl	-616(%rbp), %ecx
	testl	%ecx, %ecx
	jle	.L110
	movq	40(%rbp), %rax
	movq	(%rax), %rsi
	movq	32(%rbp), %rax
	movq	(%rax), %rcx
	movl	-616(%rbp), %eax
	leal	-1(%rax), %edx
	xorl	%eax, %eax
	addq	$1, %rdx
	salq	$4, %rdx
	.p2align 4,,10
	.p2align 3
.L111:
	movupd	(%rcx,%rax), %xmm0
	movups	%xmm0, (%rsi,%rax)
	addq	$16, %rax
	cmpq	%rax, %rdx
	jne	.L111
.L110:
	movq	-672(%rbp), %rsi
	movq	%r12, %rdi
	leaq	-312(%rbp), %r14
	xorl	%ebx, %ebx
	call	__gmpz_set@PLT
	movl	-556(%rbp), %eax
	movl	-620(%rbp), %r15d
	movq	%r14, -392(%rbp)
	leal	-1(%rax), %ecx
	movq	-600(%rbp), %rax
	addq	$1, %rcx
	salq	$4, %rcx
	leaq	(%rax,%rcx), %r13
	leaq	-320(%rbp), %rax
	movq	%r13, %r14
	movq	%rax, -336(%rbp)
	.p2align 4,,10
	.p2align 3
.L109:
	movq	40(%rbp), %rcx
	movq	%rbx, %rax
	movq	%r12, %rdi
	addq	(%rcx), %rax
	movsd	(%rax), %xmm7
	movsd	8(%rax), %xmm6
	movsd	%xmm7, -384(%rbp)
	movsd	%xmm6, -352(%rbp)
	call	__gmpz_get_d@PLT
	sqrtsd	%xmm0, %xmm5
	movq	%r12, %rdi
	movsd	%xmm0, -328(%rbp)
	movsd	%xmm5, -368(%rbp)
	call	_Z10exp_itlognP12__mpz_struct@PLT
	cmpl	$1, %r15d
	je	.L112
	movsd	-368(%rbp), %xmm5
	movsd	-384(%rbp), %xmm3
	movsd	-352(%rbp), %xmm4
	divsd	%xmm5, %xmm3
	divsd	%xmm5, %xmm4
	movapd	%xmm1, %xmm5
	movapd	%xmm3, %xmm2
	mulsd	%xmm3, %xmm1
	mulsd	%xmm0, %xmm2
	mulsd	%xmm4, %xmm5
	mulsd	%xmm4, %xmm0
	subsd	%xmm5, %xmm2
	addsd	%xmm0, %xmm1
.L113:
	movq	-400(%rbp), %rdi
	xorl	%esi, %esi
	movsd	%xmm2, -368(%rbp)
	movsd	%xmm1, -352(%rbp)
	call	mpfr_get_d@PLT
	sqrtsd	%xmm0, %xmm0
	sqrtsd	%xmm0, %xmm0
	movsd	.LC0(%rip), %xmm6
	cmpl	$5120, %r15d
	movsd	-328(%rbp), %xmm7
	movl	$5120, %eax
	divsd	%xmm0, %xmm6
	cmovbe	%r15d, %eax
	movq	%r12, %rdi
	movsd	.LC1(%rip), %xmm3
	mulsd	%xmm7, %xmm3
	movapd	%xmm6, %xmm0
	mulsd	%xmm7, %xmm0
	minsd	%xmm3, %xmm0
	cvttsd2siq	%xmm0, %rdx
	cmpl	%eax, %edx
	cmovbe	%edx, %eax
	movl	%eax, %r13d
	call	__gmpz_get_d@PLT
	call	__log_finite@PLT
	mulsd	-608(%rbp), %xmm0
	movq	-336(%rbp), %rsi
	movq	-392(%rbp), %rdi
	call	sincos@PLT
	movl	-556(%rbp), %edx
	movsd	-320(%rbp), %xmm3
	movsd	-312(%rbp), %xmm0
	movq	-600(%rbp), %rax
	movsd	-352(%rbp), %xmm1
	testl	%edx, %edx
	movsd	-368(%rbp), %xmm2
	jle	.L116
	.p2align 4,,10
	.p2align 3
.L133:
	movsd	(%rax), %xmm4
	movapd	%xmm0, %xmm5
	addq	$16, %rax
	addsd	%xmm2, %xmm4
	mulsd	%xmm1, %xmm5
	movsd	%xmm4, -16(%rax)
	movsd	-8(%rax), %xmm4
	addsd	%xmm1, %xmm4
	mulsd	%xmm3, %xmm1
	movsd	%xmm4, -8(%rax)
	movapd	%xmm0, %xmm4
	cmpq	%r14, %rax
	mulsd	%xmm2, %xmm4
	mulsd	%xmm3, %xmm2
	addsd	%xmm4, %xmm1
	subsd	%xmm5, %xmm2
	jne	.L133
.L116:
	subl	%r13d, %r15d
	movl	%r13d, %edx
	movq	%r12, %rsi
	movq	%r12, %rdi
	addq	$16, %rbx
	call	__gmpz_add_ui@PLT
	testl	%r15d, %r15d
	jne	.L109
	movq	%r12, %rdi
	call	__gmpz_clear@PLT
	jmp	.L33
	.p2align 4,,10
	.p2align 3
.L41:
	movl	-444(%rbp), %eax
	movl	$0, -444(%rbp)
	movl	%eax, 8(%rbx)
	movl	%eax, -488(%rbp)
	jmp	.L118
	.p2align 4,,10
	.p2align 3
.L155:
	movq	-360(%rbp), %rdx
	movabsq	$9223090561878065152, %rax
	movq	$1, -368(%rbp)
	orq	%rax, %rdx
	movq	%rdx, %rax
	movabsq	$-281474976710656, %rdx
	andq	%rdx, %rax
	orq	$1, %rax
	movq	%rax, -360(%rbp)
	movdqa	-368(%rbp), %xmm7
	movaps	%xmm7, -352(%rbp)
	jmp	.L84
	.p2align 4,,10
	.p2align 3
.L158:
	movq	-376(%rbp), %rdx
	movabsq	$9223090561878065152, %rax
	movq	$1, -384(%rbp)
	orq	%rax, %rdx
	movq	%rdx, %rax
	movabsq	$-281474976710656, %rdx
	andq	%rdx, %rax
	orq	$1, %rax
	movq	%rax, -376(%rbp)
	movdqa	-384(%rbp), %xmm6
	movaps	%xmm6, -432(%rbp)
	jmp	.L91
	.p2align 4,,10
	.p2align 3
.L112:
	movq	%r12, %rdi
	call	__gmpz_get_d@PLT
	sqrtsd	%xmm0, %xmm3
	movq	%r12, %rdi
	movsd	%xmm3, -352(%rbp)
	call	_Z10exp_itlognP12__mpz_struct@PLT
	movsd	-352(%rbp), %xmm3
	movapd	%xmm0, %xmm2
	divsd	%xmm3, %xmm1
	divsd	%xmm3, %xmm2
	jmp	.L113
	.p2align 4,,10
	.p2align 3
.L156:
	movq	-360(%rbp), %rdx
	movabsq	$9223090561878065152, %rax
	movabsq	$9223372036854775807, %rcx
	movq	$0, -368(%rbp)
	orq	%rax, %rdx
	movq	%rdx, %rax
	movabsq	$-281474976710656, %rdx
	andq	%rdx, %rax
	xorl	%edx, %edx
	cmpl	$-1, -232(%rbp)
	sete	%dl
	andq	%rcx, %rax
	salq	$63, %rdx
	orq	%rdx, %rax
	movq	%rax, -360(%rbp)
	movdqa	-368(%rbp), %xmm7
	movaps	%xmm7, -352(%rbp)
	jmp	.L84
	.p2align 4,,10
	.p2align 3
.L159:
	movq	-376(%rbp), %rdx
	movabsq	$9223090561878065152, %rax
	movabsq	$9223372036854775807, %rcx
	movq	$0, -384(%rbp)
	orq	%rax, %rdx
	movq	%rdx, %rax
	movabsq	$-281474976710656, %rdx
	andq	%rdx, %rax
	xorl	%edx, %edx
	cmpl	$-1, -200(%rbp)
	sete	%dl
	andq	%rcx, %rax
	salq	$63, %rdx
	orq	%rdx, %rax
	movq	%rax, -376(%rbp)
	movdqa	-384(%rbp), %xmm7
	movaps	%xmm7, -432(%rbp)
	jmp	.L91
	.p2align 4,,10
	.p2align 3
.L161:
	leaq	-240(%rbp), %rdi
	call	mpfr_clear@PLT
	leaq	-272(%rbp), %rdi
	call	mpfr_clear@PLT
	leaq	-176(%rbp), %rdi
	call	mpfr_clear@PLT
	leaq	-144(%rbp), %rdi
	call	mpfr_clear@PLT
	leaq	-208(%rbp), %rdi
	call	mpfr_clear@PLT
	jmp	.L80
	.p2align 4,,10
	.p2align 3
.L152:
	movq	-520(%rbp), %rdi
	xorl	%ecx, %ecx
	movl	$1, %esi
	movq	%rdi, %rdx
	call	mpfr_ui_div@PLT
	movq	-96(%rbp), %rsi
	movabsq	$-9223372036854775806, %rax
	cmpq	%rax, %rsi
	je	.L166
	movabsq	$-9223372036854775805, %rax
	cmpq	%rax, %rsi
	je	.L167
	movabsq	$-9223372036854775807, %rax
	cmpq	%rax, %rsi
	je	.L168
	movl	-104(%rbp), %ecx
	xorl	%edi, %edi
	movabsq	$9223372036854775807, %rdx
	cmpl	$-1, %ecx
	sete	%dil
	movq	%rdi, %rax
	movq	-632(%rbp), %rdi
	salq	$63, %rax
	andq	%rdx, %rdi
	orq	%rax, %rdi
	movq	-88(%rbp), %rax
	cmpq	$-16381, %rsi
	movq	%rdi, -632(%rbp)
	movq	8(%rax), %rdx
	movq	(%rax), %rax
	jl	.L169
	addw	$16382, %si
	movabsq	$-9223090561878065153, %r8
	andl	$32767, %esi
	andq	%r8, %rdi
	salq	$48, %rsi
	orq	%rsi, %rdi
.L67:
	leaq	(%rdx,%rdx), %rsi
	movabsq	$-281474976710656, %r8
	salq	$49, %rdx
	andq	%r8, %rdi
	shrq	$15, %rax
	shrq	$16, %rsi
	orq	%rax, %rdx
	orq	%rsi, %rdi
	movq	%rdx, -640(%rbp)
	movq	%rdi, -632(%rbp)
	movdqa	-640(%rbp), %xmm7
	movaps	%xmm7, -352(%rbp)
.L62:
	movq	-520(%rbp), %rsi
	leaq	-240(%rbp), %rdi
	xorl	%edx, %edx
	movsd	.LC0(%rip), %xmm6
	divsd	-512(%rbp), %xmm6
	movsd	%xmm6, -456(%rbp)
	call	mpfr_set4@PLT
	movl	-136(%rbp), %ecx
	leaq	-144(%rbp), %rsi
	leaq	-208(%rbp), %rdi
	xorl	%edx, %edx
	call	mpfr_set4@PLT
.L68:
	leaq	-288(%rbp), %rax
	movdqa	.LC21(%rip), %xmm0
	leaq	-80(%rbp), %rdi
	movabsq	$-9223372036854775806, %r12
	xorl	%edx, %edx
	movq	%rax, -56(%rbp)
	movq	-400(%rbp), %rax
	movq	$0, -72(%rbp)
	movq	$106, -80(%rbp)
	movaps	%xmm0, -544(%rbp)
	movl	8(%rax), %ecx
	movq	%rax, %rsi
	movl	$1, -72(%rbp)
	movq	%r12, -64(%rbp)
	call	mpfr_set4@PLT
	movq	-64(%rbp), %rcx
	cmpq	%r12, %rcx
	je	.L170
	movabsq	$-9223372036854775805, %rax
	cmpq	%rax, %rcx
	je	.L171
	movabsq	$-9223372036854775807, %rax
	cmpq	%rax, %rcx
	je	.L172
	xorl	%esi, %esi
	cmpl	$-1, -72(%rbp)
	movabsq	$9223372036854775807, %rdx
	sete	%sil
	movq	%rsi, %rax
	movq	-648(%rbp), %rsi
	salq	$63, %rax
	andq	%rdx, %rsi
	orq	%rax, %rsi
	movq	-56(%rbp), %rax
	cmpq	$-16381, %rcx
	movq	%rsi, -648(%rbp)
	movq	8(%rax), %rdx
	movq	(%rax), %rax
	jge	.L74
	movq	$-16381, %rdi
	subq	%rcx, %rdi
	xorl	%ecx, %ecx
	.p2align 4,,10
	.p2align 3
.L75:
	movq	%rdx, %rsi
	shrq	%rax
	addq	$1, %rcx
	salq	$63, %rsi
	shrq	%rdx
	orq	%rsi, %rax
	cmpq	%rcx, %rdi
	jne	.L75
	movq	-648(%rbp), %rsi
	movabsq	$-9223090561878065153, %rcx
	andq	%rcx, %rsi
.L76:
	leaq	(%rdx,%rdx), %rcx
	movabsq	$-281474976710656, %rdi
	salq	$49, %rdx
	andq	%rdi, %rsi
	shrq	$15, %rax
	shrq	$16, %rcx
	orq	%rax, %rdx
	orq	%rcx, %rsi
	movq	%rdx, -656(%rbp)
	movq	%rsi, -648(%rbp)
	movdqa	-656(%rbp), %xmm7
	movaps	%xmm7, -592(%rbp)
.L71:
	movdqa	-352(%rbp), %xmm6
	movaps	%xmm6, -576(%rbp)
	jmp	.L69
	.p2align 4,,10
	.p2align 3
.L148:
	movslq	-484(%rbp), %r12
	leaq	-272(%rbp), %rdi
	leaq	-144(%rbp), %r14
	movq	%r12, %rsi
	call	mpfr_init2@PLT
	leaq	-240(%rbp), %rdi
	movq	%r12, %rsi
	call	mpfr_init2@PLT
	leaq	-208(%rbp), %rdi
	movq	%r12, %rsi
	call	mpfr_init2@PLT
	leaq	-176(%rbp), %rdi
	movl	$53, %esi
	call	mpfr_init2@PLT
	movq	%r12, %rsi
	movq	%r14, %rdi
	call	mpfr_init2@PLT
	xorl	%esi, %esi
	movq	%r14, %rdi
	call	mpfr_const_pi@PLT
	movl	$2, %edx
	xorl	%ecx, %ecx
	movq	%r14, %rsi
	movq	%r14, %rdi
	call	mpfr_mul_si@PLT
	movl	-484(%rbp), %edx
	jmp	.L52
	.p2align 4,,10
	.p2align 3
.L149:
	movq	-472(%rbp), %rdx
	movabsq	$9223090561878065152, %rax
	movq	$1, -480(%rbp)
	orq	%rax, %rdx
	movq	%rdx, %rax
	movabsq	$-281474976710656, %rdx
	andq	%rdx, %rax
	orq	$1, %rax
	movq	%rax, -472(%rbp)
	movdqa	-480(%rbp), %xmm1
	jmp	.L54
	.p2align 4,,10
	.p2align 3
.L87:
	addw	$16382, %cx
	movabsq	$-9223090561878065153, %rdi
	andl	$32767, %ecx
	andq	%rdi, %rsi
	salq	$48, %rcx
	orq	%rcx, %rsi
	jmp	.L89
	.p2align 4,,10
	.p2align 3
.L94:
	addw	$16382, %cx
	movabsq	$-9223090561878065153, %rdi
	andl	$32767, %ecx
	andq	%rdi, %rsi
	salq	$48, %rcx
	orq	%rcx, %rsi
	jmp	.L96
.L157:
	movq	-360(%rbp), %rdx
	movabsq	$-9223090561878065153, %rax
	movabsq	$9223372036854775807, %rcx
	movq	$0, -368(%rbp)
	andq	%rax, %rdx
	movq	%rdx, %rax
	movabsq	$-281474976710656, %rdx
	andq	%rdx, %rax
	xorl	%edx, %edx
	cmpl	$-1, -232(%rbp)
	sete	%dl
	andq	%rcx, %rax
	salq	$63, %rdx
	orq	%rdx, %rax
	movq	%rax, -360(%rbp)
	movdqa	-368(%rbp), %xmm6
	movaps	%xmm6, -352(%rbp)
	jmp	.L84
.L160:
	movq	-376(%rbp), %rdx
	movabsq	$-9223090561878065153, %rax
	movabsq	$9223372036854775807, %rcx
	movq	$0, -384(%rbp)
	andq	%rax, %rdx
	movq	%rdx, %rax
	movabsq	$-281474976710656, %rdx
	andq	%rdx, %rax
	xorl	%edx, %edx
	cmpl	$-1, -200(%rbp)
	sete	%dl
	andq	%rcx, %rax
	salq	$63, %rdx
	orq	%rdx, %rax
	movq	%rax, -376(%rbp)
	movdqa	-384(%rbp), %xmm6
	movaps	%xmm6, -432(%rbp)
	jmp	.L91
.L150:
	movq	-472(%rbp), %rdx
	movabsq	$9223090561878065152, %rax
	orq	%rax, %rdx
.L137:
	movq	%rdx, %rax
	movabsq	$-281474976710656, %rdx
	movabsq	$9223372036854775807, %rcx
	andq	%rdx, %rax
	xorl	%edx, %edx
	cmpl	$-1, -104(%rbp)
	movq	$0, -480(%rbp)
	sete	%dl
	andq	%rcx, %rax
	salq	$63, %rdx
	orq	%rdx, %rax
	movq	%rax, -472(%rbp)
	movdqa	-480(%rbp), %xmm1
	jmp	.L54
.L57:
	addw	$16382, %cx
	movabsq	$-9223090561878065153, %rdi
	andl	$32767, %ecx
	andq	%rdi, %rsi
	salq	$48, %rcx
	orq	%rcx, %rsi
	jmp	.L59
.L145:
	leaq	.LC11(%rip), %rsi
	leaq	_ZSt4cout(%rip), %rdi
	movl	$59, %edx
	movq	-504(%rbp), %r12
	leaq	_ZSt4cout(%rip), %r13
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	_ZSt4cout(%rip), %rax
	movq	-24(%rax), %rax
	movq	240(%r13,%rax), %rbx
	testq	%rbx, %rbx
	je	.L46
	cmpb	$0, 56(%rbx)
	je	.L44
	movsbl	67(%rbx), %esi
.L45:
	leaq	_ZSt4cout(%rip), %rdi
	call	_ZNSo3putEc@PLT
	movq	%rax, %rdi
	call	_ZNSo5flushEv@PLT
	leaq	.LC12(%rip), %rsi
	leaq	_ZSt4cout(%rip), %rdi
	movl	$59, %edx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	_ZSt4cout(%rip), %rax
	movq	-24(%rax), %rax
	movq	240(%r13,%rax), %rbx
	testq	%rbx, %rbx
	je	.L46
	cmpb	$0, 56(%rbx)
	je	.L47
	movsbl	67(%rbx), %esi
.L48:
	leaq	_ZSt4cout(%rip), %rdi
	call	_ZNSo3putEc@PLT
	movq	%rax, %rdi
	call	_ZNSo5flushEv@PLT
	jmp	.L40
.L151:
	movq	-472(%rbp), %rdx
	movabsq	$-9223090561878065153, %rax
	andq	%rax, %rdx
	jmp	.L137
.L44:
	movq	%rbx, %rdi
	call	_ZNKSt5ctypeIcE13_M_widen_initEv@PLT
	movq	(%rbx), %rax
	leaq	_ZNKSt5ctypeIcE8do_widenEc(%rip), %rdx
	movl	$10, %esi
	movq	48(%rax), %rax
	cmpq	%rdx, %rax
	je	.L45
	movq	%rbx, %rdi
	call	*%rax
	movsbl	%al, %esi
	jmp	.L45
.L169:
	movq	$-16381, %r8
	subq	%rsi, %r8
	xorl	%esi, %esi
	.p2align 4,,10
	.p2align 3
.L66:
	movq	%rdx, %rdi
	shrq	%rax
	addq	$1, %rsi
	salq	$63, %rdi
	shrq	%rdx
	orq	%rdi, %rax
	cmpq	%rsi, %r8
	jne	.L66
	movq	-632(%rbp), %rdi
	movabsq	$-9223090561878065153, %rsi
	andq	%rsi, %rdi
	jmp	.L67
.L146:
	leaq	-80(%rbp), %rdi
	movsd	-440(%rbp), %xmm0
	call	frexp@PLT
	movl	-80(%rbp), %eax
	leal	-1(%rax), %r13d
	jmp	.L49
.L147:
	movq	%r12, -328(%rbp)
	leaq	-80(%rbp), %rdi
	movsd	-328(%rbp), %xmm0
	call	frexp@PLT
	movl	-80(%rbp), %eax
	leal	-1(%rax), %ecx
	jmp	.L50
.L170:
	movq	-648(%rbp), %rdx
	movabsq	$9223090561878065152, %rax
	movq	$1, -656(%rbp)
	orq	%rax, %rdx
	movq	%rdx, %rax
	movabsq	$-281474976710656, %rdx
	andq	%rdx, %rax
	orq	$1, %rax
	movq	%rax, -648(%rbp)
	movdqa	-656(%rbp), %xmm7
	movaps	%xmm7, -592(%rbp)
	jmp	.L71
.L166:
	movq	-632(%rbp), %rdx
	movabsq	$9223090561878065152, %rax
	movq	$1, -640(%rbp)
	movl	-104(%rbp), %ecx
	orq	%rax, %rdx
	movq	%rdx, %rax
	movabsq	$-281474976710656, %rdx
	andq	%rdx, %rax
	orq	$1, %rax
	movq	%rax, -632(%rbp)
	movdqa	-640(%rbp), %xmm6
	movaps	%xmm6, -352(%rbp)
	jmp	.L62
.L171:
	movq	-648(%rbp), %rdx
	movabsq	$9223090561878065152, %rax
	movabsq	$9223372036854775807, %rcx
	movq	$0, -656(%rbp)
	orq	%rax, %rdx
	movq	%rdx, %rax
	movabsq	$-281474976710656, %rdx
	andq	%rdx, %rax
	xorl	%edx, %edx
	cmpl	$-1, -72(%rbp)
	sete	%dl
	andq	%rcx, %rax
	salq	$63, %rdx
	orq	%rdx, %rax
	movq	%rax, -648(%rbp)
	movdqa	-656(%rbp), %xmm7
	movaps	%xmm7, -592(%rbp)
	jmp	.L71
.L167:
	movq	-632(%rbp), %rdx
	movabsq	$9223090561878065152, %rax
	orq	%rax, %rdx
.L138:
	movl	-104(%rbp), %ecx
	movq	%rdx, %rax
	movabsq	$-281474976710656, %rdx
	andq	%rdx, %rax
	xorl	%edx, %edx
	movabsq	$9223372036854775807, %rsi
	movq	$0, -640(%rbp)
	cmpl	$-1, %ecx
	sete	%dl
	andq	%rsi, %rax
	salq	$63, %rdx
	orq	%rdx, %rax
	movq	%rax, -632(%rbp)
	movdqa	-640(%rbp), %xmm6
	movaps	%xmm6, -352(%rbp)
	jmp	.L62
.L47:
	movq	%rbx, %rdi
	call	_ZNKSt5ctypeIcE13_M_widen_initEv@PLT
	movq	(%rbx), %rax
	leaq	_ZNKSt5ctypeIcE8do_widenEc(%rip), %rdx
	movl	$10, %esi
	movq	48(%rax), %rax
	cmpq	%rdx, %rax
	je	.L48
	movq	%rbx, %rdi
	call	*%rax
	movsbl	%al, %esi
	jmp	.L48
.L74:
	addw	$16382, %cx
	movabsq	$-9223090561878065153, %rdi
	andl	$32767, %ecx
	andq	%rdi, %rsi
	salq	$48, %rcx
	orq	%rcx, %rsi
	jmp	.L76
.L172:
	movq	-648(%rbp), %rdx
	movabsq	$-9223090561878065153, %rax
	movabsq	$9223372036854775807, %rcx
	movq	$0, -656(%rbp)
	andq	%rax, %rdx
	movq	%rdx, %rax
	movabsq	$-281474976710656, %rdx
	andq	%rdx, %rax
	xorl	%edx, %edx
	cmpl	$-1, -72(%rbp)
	sete	%dl
	andq	%rcx, %rax
	salq	$63, %rdx
	orq	%rdx, %rax
	movq	%rax, -648(%rbp)
	movdqa	-656(%rbp), %xmm6
	movaps	%xmm6, -592(%rbp)
	jmp	.L71
.L168:
	movq	-632(%rbp), %rdx
	movabsq	$-9223090561878065153, %rax
	andq	%rax, %rdx
	jmp	.L138
.L165:
	movl	%eax, %edi
	call	cudaGetErrorString@PLT
	leaq	.LC2(%rip), %r8
	leaq	.LC19(%rip), %rdx
	movl	$438, %r9d
	movq	%rax, %rcx
.L139:
	movq	stderr(%rip), %rdi
	leaq	.LC4(%rip), %rsi
	xorl	%eax, %eax
	call	fprintf@PLT
	movq	stderr(%rip), %rcx
	leaq	.LC5(%rip), %rdi
	movl	$22, %edx
	movl	$1, %esi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L107:
	movl	%eax, %edi
	call	cudaGetErrorString@PLT
	movl	$441, %r9d
	leaq	.LC2(%rip), %r8
	movq	%rax, %rcx
	leaq	.LC20(%rip), %rdx
	jmp	.L139
.L46:
	call	_ZSt16__throw_bad_castv@PLT
.L163:
	movl	%eax, %edi
	call	cudaGetErrorString@PLT
	movl	$428, %r9d
	leaq	.LC2(%rip), %r8
	movq	%rax, %rcx
	leaq	.LC17(%rip), %rdx
	jmp	.L139
.L164:
	movl	%eax, %edi
	call	cudaGetErrorString@PLT
	movl	$431, %r9d
	leaq	.LC2(%rip), %r8
	movq	%rax, %rcx
	leaq	.LC18(%rip), %rdx
	jmp	.L139
.L153:
	movaps	%xmm0, -352(%rbp)
	jmp	.L68
	.cfi_endproc
.LFE2888:
	.size	_Z17zeta_block_stage2P12__mpz_structjP13__mpfr_structdiPSt7complexIdEPP20precomputation_tablePP7double2S8_SB_PS5_iP11CUstream_stiP15pthread_mutex_ti, .-_Z17zeta_block_stage2P12__mpz_structjP13__mpfr_structdiPSt7complexIdEPP20precomputation_tablePP7double2S8_SB_PS5_iP11CUstream_stiP15pthread_mutex_ti
	.section	.text.startup,"ax",@progbits
	.p2align 4,,15
	.type	_GLOBAL__sub_I__Z18stage_2_block_sizedd, @function
_GLOBAL__sub_I__Z18stage_2_block_sizedd:
.LFB3371:
	.cfi_startproc
	leaq	_ZStL8__ioinit(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	_ZNSt8ios_base4InitC1Ev@PLT
	movq	_ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	leaq	__dso_handle(%rip), %rdx
	leaq	_ZStL8__ioinit(%rip), %rsi
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
	jmp	__cxa_atexit@PLT
	.cfi_endproc
.LFE3371:
	.size	_GLOBAL__sub_I__Z18stage_2_block_sizedd, .-_GLOBAL__sub_I__Z18stage_2_block_sizedd
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I__Z18stage_2_block_sizedd
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC0:
	.long	0
	.long	1072693248
	.align 8
.LC1:
	.long	2696277389
	.long	1053869815
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC13:
	.long	0
	.long	0
	.long	0
	.long	1073676288
	.section	.rodata.cst8
	.align 8
.LC14:
	.long	0
	.long	-1074790400
	.align 8
.LC15:
	.long	0
	.long	1071644672
	.section	.rodata.cst16
	.align 16
.LC21:
	.quad	-8905435550453399104
	.quad	4612128158286889681
	.hidden	__dso_handle
	.ident	"GCC: (Debian 6.3.0-18) 6.3.0 20170516"
	.section	.note.GNU-stack,"",@progbits
