#ifndef DDTABLE_H
#define DDTABLE_H

#include <stdint.h>
#include <stdbool.h>

// Grumble grumble, C++
#ifdef __cplusplus
#define restrict __restrict__
#endif

//! Default NULL value (not a value) for our table
#define DDTABLE_NULL_VAL (scalar) 0xdeadbeef

#ifdef MYRIAD_ALLOCATOR
#include "myriad_alloc.h"
#define _my_malloc(size) myriad_malloc(size, true)
#define _my_calloc(num, size) myriad_calloc(num, size, true)
#define _my_free(loc) myriad_free(loc)
#else
#define _my_malloc(size) malloc(size)
#define _my_calloc(num, size) calloc(num, size)
#define _my_free(loc) free(loc)
#endif

//! Double-to-Double function pointer for callbacks.
typedef scalar (*d2dfun) (const scalar d);

//! Hash table for scalar-valued key-value pairs.
typedef struct ddtable
{
#ifdef DEBUG
    uint64_t _ddtable_hits;
    uint64_t _ddtable_misses;
    uint64_t _ddtable_collisions;
#endif
    //! Absolute number of key-value pairs
    uint64_t num_kv_pairs;
    //! Internal size used for hashing
    uint64_t size;
    //! Fast-checker for key existence
    int_fast8_t* restrict exists;
    //! Single-alloc array for kv pairs
    scalar* restrict key_vals;
} *ddtable_t;

/**
 * @brief Initializes a new hash table with the number of keys given.
 * 
 * The number of keys given is actually a lower bound. Keys will always be
 * rounded up to the next power of 2. If the table will fit into a set of locked
 * memory pages, and the process has not exhausted its locked page limit, then
 * the table will be locked into memory, and only unlocked when freed or the
 * program exist.
 * 
 * @param num_keys Minimum number of keys to allocate the table for.
 *
 * @returns pointer to a new, initialized struct ddtable hash table
 */
extern ddtable_t ddtable_new(const uint64_t num_keys);

/**
 * @brief Frees an allocated hash table.
 *
 * If the table's pages are locked in memory, they are unlocked here.
 *
 * @param ddtable Hash table to be freed.
 */
extern void ddtable_free(ddtable_t table);

/**
 * @brief Get value corresponding to given key in the given hash table.
 * 
 * @param table Hash table to do lookup on
 * @param key Key to lookup in hash table
 * 
 * @returns corresponding value if the key exists, otherwise DDTABLE_NULL_VAL
 * 
 * @see get_check_key
 */
extern scalar ddtable_get_val(ddtable_t table,
                              const scalar key) __attribute__((pure));

/**
 * @brief Check if key-value pair has been assigned for the given key.
 *
 * @param table Hash table to do lookup on
 * @param key Key to lookup in hash table
 *
 * @returns true if key found, false otherwise
 *
 * @see get_key, get_check_key
 */
extern bool ddtable_check_key(ddtable_t table,
                              const scalar key) __attribute__((pure));

/**
 * @brief Get value, but check the key's value before returning.
 *
 * @param table Hash table to do lookup on
 * @param key Key to lookup in hash table
 *
 * @returns corresponding value if the key exists and its memcmp-identical to
 *          the given one; otherwise, DDTABLE_NULL_VAL
 *
 * @see get_val
 */
extern scalar ddtable_get_check_key(ddtable_t table,
                                    const scalar key) __attribute__((pure));

/**
 * @brief Add the given key-value pair to the given hash table.
 * 
 * @param table Hash table to do insertion on
 * @param key Key to insert into table
 * @param val Value to insert into table, corresponding to key
 *
 * @returns 0 if success, -1 if a collision occured.
 *
 * @see ddtable_clobber_val
 */
extern int ddtable_set_val(ddtable_t table, const scalar key, const scalar val);

/**
 * @brief Add the given key-value pair to the given hash table, forcefully.
 * 
 * This function will clobber any previous key-value pair on collision.
 * 
 * @param table Hash table to do insertion on
 * @param key Key to insert into table
 * @param val Value to insert into table, corresponding to key
 *
 * @see ddtable_set_val
 */
extern void ddtable_clobber_val(ddtable_t table,
                               const scalar key,
                               const scalar val);

/**
 * @brief Checks key existence before retrieving, callbacks on mismatch.
 *
 * Given a key, this function checks if it exists in the table:
 * - If it does not, it calculates the corresponding value with the callback,
 *   then adds both the key and the value to the hash table.
 * - If it does exist, the retrieved key is checked. If it matches the given
 *   key, the corresponding value is returned. Otherwise, the result of the
 *   function callback is returned, and the existing key-value pair remains.
 *
 * NOTE: Callback function must have NO side-effects.
 *
 * @param table Hash table to operate on
 * @param cb Side-effect-free function callback to calculate value from key
 * @param key Key to retrieve value from table with
 *
 * @returns value corresponding to key, either retrieved or calculated
 *
 * @see get_check_key
 * @see d2dfun
 */
extern scalar ddtable_check_get_set(ddtable_t table,
                                    const scalar key,
                                    d2dfun cb);
#undef restrict
#endif
