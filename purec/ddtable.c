/**
 * @file   ddtable.c
 * @author Pedro Rittner
 * @date   Feb 17 2015
 * @brief  Implementation of hash table for Myriad using Spooky C hash.
 *
 * Copyright (c) 2015 Pedro Ritter <pr273@cornell.edu>
 * 
 * This file is free software: you may copy, redistribute and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 * 
 * This file is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 * This file incorporates work covered by the following copyright and  
 * permission notice:
 *
 * SpookyHash: a 128-bit noncryptographic hash function
 * By Bob Jenkins, public domain
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ddtable.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <sys/resource.h>

//! Saves/shows information on collisions
#ifndef DEBUG
#define DEBUG 0
#endif

//! Checks if x is a power of 2
#define IS_POW2(x) ((x != 0) && ((x & (~x + 1)) == x))

//! Sets the seed we pass to spooky
#ifndef SPOOKY_HASH_SEED
#define SPOOKY_HASH_SEED 0
#endif

#define SC_NUMVARS		12

/**
 * @brief Special constant for spookyhash.
 *
 * SC_CONST is a constant which:
 *  - is not zero
 *  - is odd
 *  - is a not-very-regular mix of 1's and 0's
 *  - does not need any other special mathematical properties
 */
#define SC_CONST 0xdeadbeefdeadbeefLL

#define rot64(x, k)	(uint64_t) (((uint64_t) x << (int) k) | \
                                ((uint64_t) x >> (64 - (int) k)))

/**
 * @brief Short mix used in Spooky short hash.
 *
 * The goal is for each bit of the input to expand into 128 bits of
 * apparent entropy before it is fully overwritten.
 * n trials both set and cleared at least m bits of h0 h1 h2 h3
 * n: 2   m: 29
 * n: 3   m: 46
 * n: 4   m: 57
 * n: 5   m: 107
 * n: 6   m: 146
 * n: 7   m: 152
 * when run forwards or backwards
 * for all 1-bit and 2-bit diffs
 * with diffs defined by either xor or subtraction
 * with a base of all zeros plus a counter, or plus another bit, or random
*/
static inline void short_mix(uint64_t *h0,
                             uint64_t *h1,
                             uint64_t *h2,
                             uint64_t *h3)
{
	*h2 = rot64(*h2, 50);	*h2 += *h3;  *h0 ^= *h2;
	*h3 = rot64(*h3, 52);	*h3 += *h0;  *h1 ^= *h3;
	*h0 = rot64(*h0, 30);	*h0 += *h1;  *h2 ^= *h0;
	*h1 = rot64(*h1, 41);	*h1 += *h2;  *h3 ^= *h1;
	*h2 = rot64(*h2, 54);	*h2 += *h3;  *h0 ^= *h2;
	*h3 = rot64(*h3, 48);	*h3 += *h0;  *h1 ^= *h3;
	*h0 = rot64(*h0, 38);	*h0 += *h1;  *h2 ^= *h0;
	*h1 = rot64(*h1, 37);	*h1 += *h2;  *h3 ^= *h1;
	*h2 = rot64(*h2, 62);	*h2 += *h3;  *h0 ^= *h2;
	*h3 = rot64(*h3, 34);	*h3 += *h0;  *h1 ^= *h3;
	*h0 = rot64(*h0, 5);	*h0 += *h1;  *h2 ^= *h0;
	*h1 = rot64(*h1, 36);	*h1 += *h2;  *h3 ^= *h1;
}

/**
 * @brief Mix all 4 inputs together so that h0, h1 are a hash of them all.
 *
 * For two inputs differing in just the input bits where "differ" means xor
 * or subtraction, and the base value is random, or a counting value starting
 * at that bit.
 *
 * The final result will have each bit of h0, h1 flip:
 * - For every input bit, with probability 50 +- .3%
 * - For every pair of input bits, with probability 50 +- .75%
*/
static inline void short_end(uint64_t *h0,
                             uint64_t *h1,
                             uint64_t *h2,
                             uint64_t *h3)
{
	*h3 ^= *h2;  *h2 = rot64(*h2, 15);  *h3 += *h2;
	*h0 ^= *h3;  *h3 = rot64(*h3, 52);  *h0 += *h3;
	*h1 ^= *h0;  *h0 = rot64(*h0, 26);  *h1 += *h0;
	*h2 ^= *h1;  *h1 = rot64(*h1, 51);  *h2 += *h1;
	*h3 ^= *h2;  *h2 = rot64(*h2, 28);  *h3 += *h2;
	*h0 ^= *h3;  *h3 = rot64(*h3, 9);   *h0 += *h3;
	*h1 ^= *h0;  *h0 = rot64(*h0, 47);  *h1 += *h0;
	*h2 ^= *h1;  *h1 = rot64(*h1, 54);  *h2 += *h1;
	*h3 ^= *h2;  *h2 = rot64(*h2, 32);  *h3 += *h2;
	*h0 ^= *h3;  *h3 = rot64(*h3, 25);  *h0 += *h3;
	*h1 ^= *h0;  *h0 = rot64(*h0, 63);  *h1 += *h0;
}


void spooky_shorthash(const void *message,
                      size_t length,
                      uint64_t *hash1,
                      uint64_t *hash2)
{
	union
	{
		const uint8_t *p8;
		uint32_t *p32;
		uint64_t *p64;
		size_t i;
	} u;
	size_t remainder;
	uint64_t a, b, c, d;
	u.p8 = (const uint8_t *)message;

	remainder = length % 32;
	a = *hash1;
	b = *hash2;
	c = SC_CONST;
	d = SC_CONST;

	if (length > 15)
	{
		const uint64_t *endp = u.p64 + (length/32)*4;

		// handle all complete sets of 32 bytes
		for (; u.p64 < endp; u.p64 += 4)
		{
			c += u.p64[0];
			d += u.p64[1];
			short_mix(&a, &b, &c, &d);
			a += u.p64[2];
			b += u.p64[3];
		}

		// Handle the case of 16+ remaining bytes.
		if (remainder >= 16)
		{
			c += u.p64[0];
			d += u.p64[1];
			short_mix(&a, &b, &c, &d);
			u.p64 += 2;
			remainder -= 16;
		}
	}

	// Handle the last 0..15 bytes, and its length
	d = ((uint64_t)length) << 56;
	switch (remainder)
	{
		case 15:
			d += ((uint64_t)u.p8[14]) << 48;
		case 14:
			d += ((uint64_t)u.p8[13]) << 40;
		case 13:
			d += ((uint64_t)u.p8[12]) << 32;
		case 12:
			d += u.p32[2];
			c += u.p64[0];
			break;
		case 11:
			d += ((uint64_t)u.p8[10]) << 16;
		case 10:
			d += ((uint64_t)u.p8[9]) << 8;
		case 9:
			d += (uint64_t)u.p8[8];
		case 8:
			c += u.p64[0];
			break;
		case 7:
			c += ((uint64_t)u.p8[6]) << 48;
		case 6:
			c += ((uint64_t)u.p8[5]) << 40;
		case 5:
			c += ((uint64_t)u.p8[4]) << 32;
		case 4:
			c += u.p32[0];
			break;
		case 3:
			c += ((uint64_t)u.p8[2]) << 16;
		case 2:
			c += ((uint64_t)u.p8[1]) << 8;
		case 1:
			c += (uint64_t)u.p8[0];
			break;
		case 0:
			c += SC_CONST;
			d += SC_CONST;
	}
	short_end(&a, &b, &c, &d);
	*hash1 = a;
	*hash2 = b;
}

static inline uint64_t spooky_hash64(const void *message,
                                     size_t length,
                                     uint64_t seed)
{
	uint64_t hash1 = seed;
	spooky_shorthash(message, length, &hash1, &seed);
	return hash1;
}

//! Hash function using spooky 64-bit hash
static inline uint64_t dd_hash(const double key,
                               const uint64_t size)
{
#if DEBUG > 1
    printf("key: %f, ", key);
#endif
    // Can use faster & instead of % if we enforce power of 2 size.
    return spooky_hash64(&key, sizeof(double), SPOOKY_HASH_SEED) & size;
}

//! Gets the next power of two from the given number (e.g. 30 -> 32)
static uint64_t next_power_of_2(uint64_t n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n++;
    return n;
}

ddtable_t ddtable_new(const uint64_t num_keys)
{
    // Set the absolute number of key-value pairs, and also
    // set the internal size since we enforce "power of 2"-sized tables.
    const uint64_t pot_num_keys = next_power_of_2(num_keys);
    
    // Calculate the memory footprint so we can try allocate.
    const size_t pot_size = (sizeof(double) * pot_num_keys * 2) +
        sizeof(struct ddtable) + (sizeof(int_fast8_t) * pot_num_keys);

    // Allocate entire table at one time
    ddtable_t new_ht = (ddtable_t) _my_malloc(pot_size);

    // Reset entire memory region
    memset(new_ht, 0, pot_size);

    // Initialize debug information, if specified
#if DEBUG
    new_ht->_ddtable_hits = 0;
    new_ht->_ddtable_misses = 0;
    new_ht->_ddtable_collisions = 0;
#endif

    // This minus one trick is necessary for &: http://goo.gl/FlcEb0
    new_ht->size = pot_num_keys - 1;
    new_ht->num_kv_pairs = new_ht->size + 1;

    // Bitmap array starts right at the end of the struct
    new_ht->exists = (int_fast8_t*) (((intptr_t) &new_ht->exists)
                                     + sizeof(int_fast8_t*) + sizeof(double*));
    
    // Key-value pair array starts after the bitmap
    new_ht->key_vals = (double*) (((intptr_t) &new_ht->exists)
                                  + sizeof(int_fast8_t*)
                                  + sizeof(double*)
                                  + ((new_ht->size + 1) * sizeof(int_fast8_t)));

    return new_ht;
}

void ddtable_free(ddtable_t table)
{
    if (table != NULL)
    {
#if DEBUG
        // Print out debugging information for table lifetime.
        printf("table %p meta-info:\n", (void*) table);
        printf("\thits: %" PRIu64 "\n", table->_ddtable_hits);
        printf("\tmisses: %" PRIu64 "\n", table->_ddtable_misses);
        printf("\tcollisions: %" PRIu64 "\n", table->_ddtable_collisions);
#endif
        _my_free(table);
    } else {
        fprintf(stderr, "Attempt to free NULL ddtable.");
        exit(EXIT_FAILURE);
    }
}

double ddtable_get_val(ddtable_t table, const double key)
{
    const uint64_t indx = dd_hash(key, table->size);

#if DEBUG
    // Return the value, if the key exists in the table, mark as a hit.
    // Otherwise, mark it as a miss and return DDTABLE_NULL_VAL
    const bool exists = table->exists[indx];
    if (exists) // Cache hit
    {
        table->_ddtable_hits++;
        return table->key_vals[(indx << 1) + 1];
    } else { // Cache miss
        table->_ddtable_misses++;
        return DDTABLE_NULL_VAL;
    }
#else
    // Return the value, if the key exists in the table
    return (table->exists[indx]) ?
        table->key_vals[(indx << 1) + 1] : DDTABLE_NULL_VAL;
    #endif
}

double ddtable_get_check_key(ddtable_t table, const double key)
{
    const uint64_t indx = dd_hash(key, table->size);

#if DEBUG
    const bool found = table->exists[indx];
    const bool matches = table->key_vals[indx << 1] == key;
    if (!found) // Cache miss
    {
        table->_ddtable_misses++;
        return DDTABLE_NULL_VAL;
    } else if (matches) { // Cache hit
        table->_ddtable_hits++;
        return table->key_vals[(indx << 1) + 1];
    } else { // Cache collision
        table->_ddtable_collisions++;
        return DDTABLE_NULL_VAL;
    }
#else
    // If the key exists AND it's identical to the given one, return value.
    return (table->exists[indx] && table->key_vals[indx << 1] == key)
        ? table->key_vals[(indx << 1) + 1] : DDTABLE_NULL_VAL;
    #endif
}

int ddtable_set_val(ddtable_t table, const double key, const double val)
{
    const uint64_t indx = dd_hash(key, table->size);

    if(table->exists[indx])
    {
#if DEBUG > 1
        printf("ddtable_set_val: (b: %lu) collision (%f,%f) vs (%f,%f)\n",
               indx,
               key, val,
               table->key_vals[indx << 1], table->key_vals[(indx << 1) + 1]);
#endif // DEBUG > 1
#if DEBUG
        table->_ddtable_collisions++;
#endif // DEBUG
        return -1; // Collision
    } else {
        table->exists[indx] = (int_fast8_t) 1;
        table->key_vals[indx << 1] = key;
        table->key_vals[(indx << 1) + 1] = val;
        return 0;
    }
}

void ddtable_clobber_val(ddtable_t table, const double key, const double val)
{
    const uint64_t indx = dd_hash(key, table->size);
    table->exists[indx] = (int_fast8_t) 1;
    table->key_vals[indx << 1] = key;
    table->key_vals[(indx << 1) + 1] = val;
}

bool ddtable_check_key(ddtable_t table, const double key)
{
    return (bool) table->exists[dd_hash(key, table->size)];
}

double ddtable_check_get_set(ddtable_t table, const double key, d2dfun cb)
{
    const uint64_t indx = dd_hash(key, table->size);
    const uint64_t indx_s = indx << 1;
    if (table->exists[indx])
    {
        const double val = table->key_vals[indx_s + 1];
        if (val == DDTABLE_NULL_VAL || table->key_vals[indx_s] != key)
        {
#if DEBUG
            table->_ddtable_collisions++;
#endif
            return cb(key);
        } else {
#if DEBUG
            table->_ddtable_hits++;
#endif            
            return val;
        }
    } else {
#if DEBUG
        table->_ddtable_misses++;
#endif          
        const double val = cb(key);
        table->exists[indx] = (int_fast8_t) 1;
        table->key_vals[indx_s] = key;
        table->key_vals[indx_s + 1] = val;
        return val;
    }
}

#undef DEBUG
