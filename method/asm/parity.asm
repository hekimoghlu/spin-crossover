; x86-64 (System V) NASM routines used by scodesign via ctypes.
;
; double-precision Pauli-Z parity expectations from measured histograms:
;   void parity_expect(const double *p, int64 nstates, const uint32 *masks,
;                      int64 nmasks, int64 nrows, double *out)
;   out[r*nmasks + k] = sum_x p[r*nstates + x] * (-1)^popcount(x & masks[k])
;
; seniority-zero basis enumeration via popcount:
;   int64 enum_popcount(int64 nbits, int64 k, int64 *out)
;   writes every x in [0, 2^nbits) with popcount(x) == k, ascending; returns count.
;
; QAOA bitstring cost evaluation (weighted Hamming sum):
;   void weighted_bitsum(int64 nbits, const double *w, double *out)
;   out[z] = sum_q w[q] * bit_q(z) for z in [0, 2^nbits), qubit q = bit (nbits-1-q)
;
; Requires the POPCNT instruction (SSE4.2-era x86-64).

default rel
section .text
global parity_expect
global enum_popcount
global weighted_bitsum

parity_expect:
        push    rbx
        push    r12
        push    r13
        push    r14
        push    r15
        xor     r10, r10                ; r = 0
.row_loop:
        cmp     r10, r8
        jge     .done
        mov     rax, r10
        imul    rax, rsi
        lea     r11, [rdi + rax*8]      ; row pointer
        xor     r12, r12                ; k = 0
.mask_loop:
        cmp     r12, rcx
        jge     .next_row
        mov     r13d, dword [rdx + r12*4]
        xorpd   xmm0, xmm0
        xor     r14, r14                ; x = 0
.state_loop:
        cmp     r14, rsi
        jge     .store
        movsd   xmm1, qword [r11 + r14*8]
        mov     eax, r14d
        and     eax, r13d
        popcnt  eax, eax
        test    eax, 1
        jz      .plus
        subsd   xmm0, xmm1
        jmp     .cont
.plus:
        addsd   xmm0, xmm1
.cont:
        inc     r14
        jmp     .state_loop
.store:
        mov     rax, r10
        imul    rax, rcx
        add     rax, r12
        movsd   qword [r9 + rax*8], xmm0
        inc     r12
        jmp     .mask_loop
.next_row:
        inc     r10
        jmp     .row_loop
.done:
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rbx
        ret

enum_popcount:
        mov     rcx, rdi
        mov     r8, 1
        shl     r8, cl                  ; r8 = 2^nbits
        xor     rax, rax                ; count
        xor     r9, r9                  ; x
.e_loop:
        cmp     r9, r8
        jge     .e_done
        popcnt  r10, r9
        cmp     r10, rsi
        jne     .e_next
        mov     qword [rdx + rax*8], r9
        inc     rax
.e_next:
        inc     r9
        jmp     .e_loop
.e_done:
        ret

weighted_bitsum:
        mov     rcx, rdi
        mov     r8, 1
        shl     r8, cl                  ; 2^nbits
        xor     r9, r9                  ; z
.w_z:
        cmp     r9, r8
        jge     .w_done
        xorpd   xmm0, xmm0
        xor     r10, r10                ; q
.w_q:
        cmp     r10, rdi
        jge     .w_store
        mov     r11, rdi
        sub     r11, 1
        sub     r11, r10                ; bit index nbits-1-q
        bt      r9, r11
        jnc     .w_skip
        addsd   xmm0, qword [rsi + r10*8]
.w_skip:
        inc     r10
        jmp     .w_q
.w_store:
        movsd   qword [rdx + r9*8], xmm0
        inc     r9
        jmp     .w_z
.w_done:
        ret

section .note.GNU-stack noalloc noexec nowrite progbits
