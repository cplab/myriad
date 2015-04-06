/**
 * @file mmq.h
 * @author Pedro Rittner
 * @date Sat 2014-12-06 22:18:58 EST
 * @brief Common message queue macros and functions for Myriad
 */

#ifndef MMQ_H
#define MMQ_H

#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/time.h>
#include <mqueue.h>

//! Message queue name. As per POSIX.2001, it *must* start with a /
#ifndef MMQ_FNAME
#define MMQ_FNAME "/myriad_mq"
#endif

//! Maximum number of messages in the message queue.
#ifndef MMQ_MAX_MSGS
#define MMQ_MAX_MSGS 8
#endif

//! Size of each message in the message queue.
#ifndef MMQ_MSG_SIZE
#define MMQ_MSG_SIZE sizeof(uint64_t)
#endif

//! Permissions flags for message queue.
#ifndef MMQ_PERMS
#define MMQ_PERMS S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP
#endif

//! UDS name. As per POSIX.2001, it *must* start with a ./
#ifndef MMQ_UNSOCK_NAME
#define MMQ_UNSOCK_NAME "./myriad_socket"
#endif

//! @TODO
struct mmq_connector
{
    //! Myriad IPC message queue object
    mqd_t msg_queue;
    int socket_fd;
    int connection_fd;
    bool server;
};

/**
 * @brief Initializes message queue.
 *
 * @returns result of mq_open (on error, mq_open() returns (mqdt_t) -1).
 */
extern mqd_t mmq_init_mq(
    #ifndef __cplusplus
    void
    #endif
    );

/**
 * @brief Requests object to be sent over the UDS connector with timeout.
 *
 * The request is sent over a message queue, while the data is sent over a UDS.
 *
 * This function will block the execution thread until a connection is made,
 * or the socket timeout is reached, whichever comes first.
 *
 * @param connector Myriad message queue connector object pointer.
 * @param dest Destination that data will be copied to.
 * @param len Amount of data to be copied.
 *
 * @returns number of bytes read if successful, -1 otherwise
 */
extern ssize_t mmq_request_data(struct mmq_connector* connector,
                                void* dest,
                                const size_t len)
    __attribute__((nonnull(1,2)));

/**
 * @brief Initialize server/client socket with given optional timeout.
 *
 * @param server Denotes a server or client socket to be created
 * @param timeout Optional timeout of the socket on send/recieve.
 *
 * @returns socket file descriptor on success, -1 otherwise
 */
extern int mmq_socket_init(const bool server, struct timeval* timeout);

/**
 * @brief Sends data over the connector UDS with timeout.
 *
 * This function will block until a message is recieved, then the data is
 * sent over a UDS. 
 *
 * This function will block the execution thread until a connection is made,
 * the transfer is completed, or the socket timeout is reached, whichever
 * comes first.
 *
 * @param connector Myriad message queue connector object pointer.
 * @param source Location of the data to be sent.
 * @param len Amount of data to be copied.
 *
 * @returns number of bytes sent if successful, -1 otherwise
 */
extern ssize_t mmq_send_data(struct mmq_connector* connector,
                             void* source,
                             const size_t len)
    __attribute__((nonnull(1,2)));

#endif  /* MMQ_H */
