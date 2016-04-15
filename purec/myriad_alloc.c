/**
 * @file myriad_alloc.c
 * @author Pedro Rittner
 * @date Mar 6 2015
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <errno.h>
#include <sys/mman.h>
#include <semaphore.h>

#include "myriad_alloc.h"

#ifdef DEBUG
#include <assert.h>
#define SEMA_P assert(0 == sem_wait(&myriad_memdat.sema))
#define SEMA_V assert(0 == sem_post(&myriad_memdat.sema))
#else
#define SEMA_P sem_wait(&myriad_memdat.sema)
#define SEMA_V sem_post(&myriad_memdat.sema)
#endif

int myriad_alloc_init(const size_t heap_size, const size_t num_allocs)
{
    // Initialize semaphore first.
    if (sem_init(&myriad_memdat.sema, 0, 1))
    {
        perror("myriad_alloc_init, semaphore initialization: ");
        return -1;
    }

    // Start metadata index and offset at 0
    myriad_memdat.meta_indx = 0;
    myriad_memdat.offset = 0;
    
    // Set initial sizes
    myriad_memdat.heap_size = heap_size * sizeof(char);
    myriad_memdat.metadata_size = num_allocs * sizeof(struct alloc_data);

    // Initialize metadata
    myriad_memdat.metadata = calloc(num_allocs, sizeof(struct alloc_data));
    if (myriad_memdat.metadata == NULL)
    {
        perror("myriad_alloc_init, allocate metadata: ");
        exit(EXIT_FAILURE);
    }

    // Initialize heap
    myriad_memdat.heap = calloc(heap_size, sizeof(char));
    if (myriad_memdat.heap == NULL)
    {
        perror("myriad_alloc_init, allocate heap: ");
        exit(EXIT_FAILURE);
    }

    return 0;
}

void* myriad_malloc(const size_t size, bool lock)
{
    SEMA_P;
    
    /* We fail if at least one of the following holds:
     * 1) The total size of the allocation is larger than the heap size
     * 2) The size of the next allocation causes us to overrun the heap
     * 3) We have run out of places to store metadata (too many allocations)
     */
    if (size > myriad_memdat.heap_size ||
        myriad_memdat.offset + size >= myriad_memdat.heap_size ||
        myriad_memdat.meta_indx >= myriad_memdat.metadata_size)
    {
        errno = ENOMEM;
        SEMA_V;
        return NULL;
    }

    // Find next free location, set metadata.
    // Get current location in heap, which we assume to be empty.
    void* loc = &myriad_memdat.heap[myriad_memdat.offset];
    
    // Register offset and size in metadata index.
    myriad_memdat.metadata[myriad_memdat.meta_indx].offset = myriad_memdat.offset;
    myriad_memdat.metadata[myriad_memdat.meta_indx].nbytes = size;
    
    // Try to lock, if asked
    if (lock == false || mlock(loc, size) != 0)
    {
        // Silently fail
        myriad_memdat.metadata[myriad_memdat.meta_indx].memlocked = false;
    } else {
        myriad_memdat.metadata[myriad_memdat.meta_indx].memlocked = true;
    }

    // Increment index and offset to reflect new allocation
    myriad_memdat.meta_indx++;
    myriad_memdat.offset += size;

    // Return allocated region of memory
    SEMA_V;
    return loc;
}

void* myriad_calloc(const size_t num_elems, const size_t size, bool lock)
{
    void* loc = myriad_malloc(num_elems * size, lock);

    // Clear memory, if allocated.
    if (loc != NULL)
    {
        memset(loc, 0, num_elems * size);
    }

    return loc;
}

int myriad_finalize()
{
    SEMA_P;

    free(myriad_memdat.heap);
    free(myriad_memdat.metadata);

    return sem_destroy(&myriad_memdat.sema);
}

void myriad_free(void* loc)
{
    SEMA_P;
    // If the memory location is not within our bounds, panic!
    if ((uintptr_t) loc < (uintptr_t) myriad_memdat.heap ||
        (uintptr_t) loc > (uintptr_t) myriad_memdat.heap + myriad_memdat.offset)
    {
    failure:
        errno = EFAULT;
        perror("myriad_free: ");
        SEMA_V;
        exit(EXIT_FAILURE);
    }

    // Calculate offset from pointer location to heap.
    const ptrdiff_t offset = (uintptr_t) loc - (uintptr_t) myriad_memdat.heap;
    for (uint_fast32_t i = 0; i < myriad_memdat.metadata_size; i++)
    {
        if (offset == myriad_memdat.metadata[i].offset)
        {
            myriad_memdat.metadata[i].offset = 0;
            if (myriad_memdat.metadata[i].memlocked)
            {
                munlock(loc, myriad_memdat.metadata[i].nbytes);
            }
            myriad_memdat.metadata[i].memlocked = false;
            myriad_memdat.metadata[i].nbytes = 0;
            myriad_memdat.meta_indx = i;
            SEMA_V;
            return;
        }
    }

    // Something bad has happened
    goto failure;
}

#undef SEMA_P
#undef SEMA_V
