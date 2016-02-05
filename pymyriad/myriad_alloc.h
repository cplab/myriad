/**
 * @file myriad_alloc.h
 * @author Pedro Rittner
 * @brief Generic memory allocator for Myriad simulations.
 */
#ifndef MYRIAD_ALLOC_H
#define MYRIAD_ALLOC_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <semaphore.h>

/**
 * Allocation metadata structure.
 */
struct alloc_data
{
    //! Number of bytes allocated.
    size_t nbytes;
    //! Offset from starting address.
    ptrdiff_t offset;
    //! Memlocked status for this allocation.
    bool memlocked;
};

/**
 * Shared memory buffer structure.
 * Holds heap and metadata necessary for managing the heap.
 */
struct alloc_buffer
{
    //! Synchronization semaphore.
    sem_t sema;
    //! Current heap pointer offset.
    ptrdiff_t offset;
    //! Current heap size
    size_t heap_size;
    //! Current metadata size
    size_t metadata_size;
    //! Current metadata entry.
    uint64_t meta_indx;
    //! Allocation metadata.
    struct alloc_data* metadata;
    //! Raw data buffer
    char* heap;
} myriad_memdat;

/**
 * @brief Initializes memory subsystem.
 *
 * @param heap_size Initial heap size to allocate.
 * @param num_allocs Number of metadata allocations to create.
 *
 * @returns 0 if successful, -1 otherwise.
 */
extern int myriad_alloc_init(const size_t heap_size, const size_t num_allocs)
    __attribute__((cold));

/**
 * @brief Deallocates all memory.
 *
 * @returns 0 if successful, -1 otherwise.
 */
extern int myriad_finalize(void) __attribute__((cold));

/**
 * @brief Allocates a region of memory, locking it if possible
 *
 * @param size Size of allocation
 * @param lock Request to lock the allocation
 *
 * @returns Pointer to allocated memory region, or NULL on failure.
 */
extern void* myriad_malloc(const size_t size, bool lock)
    __attribute__((alloc_size(1))) __attribute__((malloc));

/**
 * @brief Allocates (optionally locked) memory with `num_elems` of size `size`.
 * 
 * @param num_elems Number of elements to allocate
 * @param size Size of each element to allocate
 * @param lock Request to lock the allocation
 *
 * @returns Pointer to allocated memory region, or NULL on failure.
 */
extern void* myriad_calloc(const size_t num_elems, const size_t size, bool lock)
    __attribute__((alloc_size(1,2))) __attribute__((malloc));

/**
 * @brief Frees a section of memory allocated by myriad_[m|c]alloc
 * 
 * @param loc Location of memory to free
 */
extern void myriad_free(void* loc) __attribute__((nonnull));

#endif
