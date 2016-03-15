/**
 * @file myriad_communicator.h
 * @author Pedro Rittner
 * @date Feb 2016-02-22 21:28:58 EST
 * @brief Communication between C and CPython via UDS object
 */

#ifndef MYRIAD_COMMUNICATOR_H
#define MYRIAD_COMMUNICATOR_H

#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/time.h>

//! UDS name. As per POSIX.2001, it *must* start with a ./
#ifndef UNSOCK_NAME
#define UNSOCK_NAME "./myriad_socket"
#endif

//! Initializes the socket in server mode and returns its file descriptor
extern int m_server_socket_init(const int num_conns);

//! Accepts a connection via a socket initialized in server mode
extern int m_server_socket_accept(int socket_fd);

//! Initializes the socket in client mode and returns its file descriptor
extern int m_client_socket_init(void);

//! Receives data across the socket using the provided socket file descriptor
extern ssize_t m_receive_data(int socket_fd, void *dest, const size_t len)
    __attribute__((nonnull(2)));

//! Receives a single integer value, parsed from the socket
extern int m_receive_int(int socket_fd, int* dest)
    __attribute__((nonnull(2)));

//! Sends a single integer value as a fixed-length character buffer
extern int m_send_int(int socket_fd, int src);

//! Sends data across the socket using the provided connection file descriptor
extern ssize_t m_send_data(int socket_fd, void const* source, const size_t len)
    __attribute__((nonnull(2)));

//! Terminates the socket
extern int m_close_socket(int socket_fd);

#endif  /* MYRIAD_COMMUNICATOR_H */
