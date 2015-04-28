#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include <errno.h>

#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <mqueue.h>

#include "mmq.h"

//! Universal address struct
struct sockaddr_un address = {.sun_family = AF_UNIX,
                              .sun_path = MMQ_UNSOCK_NAME};

mqd_t mmq_init_mq(void)
{
    // Set message queue attributes
    struct mq_attr attrs;
    attrs.mq_maxmsg = MMQ_MAX_MSGS;   // Max. # of messages on queue 
    attrs.mq_msgsize = MMQ_MSG_SIZE;  // Max. message size (bytes)

    // Create queue with options and return
    mqd_t queue = mq_open(MMQ_FNAME,
                          O_RDWR | O_CREAT | O_EXCL,
                          MMQ_PERMS,
                          &attrs);
    if (queue == -1)
    {
        perror("mmq_init_mq: mq_open failed");
        return -1;
    } else {
        return queue;
    }
}

int mmq_socket_init(const bool server, struct timeval* timeout)
{
    int socket_fd = socket(PF_UNIX, SOCK_STREAM, 0);
    if(socket_fd < 0)
    {
        perror("mmq_socket_init: socket() failed");
        return -1;
    }
    
    if (server)
    {
        // Try setting timeout if provided.
        if (timeout != NULL)
        {
            if (setsockopt(socket_fd,
                           SOL_SOCKET,
                           SO_SNDTIMEO,
                           timeout,
                           sizeof(struct timeval)) == -1)
            {
                perror("mmq_socket_init: setockopt timeout");
                return -1;
            }
        }

        if(bind(socket_fd,
                (struct sockaddr*) &address, 
                sizeof(struct sockaddr_un)) != 0)
        {
            perror("mmq_socket_init: bind() failed");
            return -1;
        }

        //TODO: Get rid of magic number for backlog
        if(listen(socket_fd, 5) != 0)
        {
            perror("mmq_socket_init: listen() failed");
            return -1;
        }
    } else {
        // Try setting timeout if provided.
        if (timeout != NULL)
        {
            if (setsockopt(socket_fd,
                           SOL_SOCKET,
                           SO_RCVTIMEO,
                           timeout,
                           sizeof(struct timeval)) == -1)
            {
                perror("mmq_socket_init: setockopt timeout");
                return -1;
            }
        }
    }

    return socket_fd;
}


ssize_t mmq_send_data(struct mmq_connector* connector,
                      void* source,
                      const size_t len)
{
    socklen_t address_length __attribute__((unused));
    ssize_t num_bytes_sent = -1, total_bytes = 0;
    if (connector->connection_fd == -1)
    {
        if((connector->connection_fd = accept(connector->socket_fd,
                                              (struct sockaddr *) &address,
                                              &address_length)) == -1)
        {
            perror("mmq_send_data: accept() failed");
            return -1;
        }
    }

    // If request is too large, do muliple writes until completed
    do
    {
        num_bytes_sent = write(connector->connection_fd,
                               source,
                               len - total_bytes);
        if (num_bytes_sent <= 0)
        {
            perror("mmq_send_data: UDS write() failed");
            return -1;
        }
        total_bytes += num_bytes_sent;
    } while(total_bytes != (ssize_t) len);

    return total_bytes;
}

ssize_t mmq_request_data(struct mmq_connector* connector,
                         void* dest,
                         const size_t len)
{
    // Blocks here until connection is established
    if (connector->connection_fd == -1)
    {
        if ((connector->connection_fd = connect(connector->socket_fd, 
                                                (struct sockaddr*) &address, 
                                                sizeof(struct sockaddr_un))) == -1)
        {
            perror("mmq_request_data: connect() failed");
            return -1;
        }
    }
    
    ssize_t num_bytes_read = -1, total_bytes = 0;
    do
    {
        num_bytes_read = read(connector->socket_fd,
                              dest,
                              len - total_bytes);
        if (num_bytes_read <= 0)
        {
            perror("mmq_request_data: read() failed");
            return -1;
        }
        total_bytes += num_bytes_read;
    } while((size_t) total_bytes != len);
    
    return total_bytes;
}
