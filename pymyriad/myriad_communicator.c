#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdarg.h>
#include <signal.h>
#include <string.h>
#include <iso646.h>

#include <unistd.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/types.h>

#include "myriad_communicator.h"

//! Socket address, used for initializing the socket
static const struct sockaddr_un SOCK_ADDRESS = {.sun_family = AF_UNIX,
                                                .sun_path = UNSOCK_NAME};

#ifndef INT_BUFF_LEN
#define INT_BUFF_LEN sizeof(int)
#endif

#ifdef DEBUG
#define debug_puts(msg, fp) fputs(msg, fp)
#else
#define debug_puts(msg, fp) do {} while(0)
#endif

#ifdef __STDC_LIB_EXT1__
#define safe_sscanf(buff, msg, ...) sscanf_s(buff, msg, __VA_ARGS__)
#else
#define safe_sscanf(buff, msg, ...) sscanf(buff, msg, __VA_ARGS__)
#endif  // __STDC_LIB_EXT1__


int m_server_socket_init(const int num_conns)
{
    int socket_fd = socket(PF_UNIX, SOCK_STREAM, 0);
    if(socket_fd < 0)
    {
        perror("m_server_socket_init: socket() failed");
        return -1;
    }

    // Bind the socket to the UDS address
    if (bind(socket_fd,
             (const struct sockaddr*) &SOCK_ADDRESS,
             sizeof(struct sockaddr_un)))
    {
        perror("m_server_socket_init: bind() failed");
        return -1;
    }
    
    // Attempt to listen for connections
    if (listen(socket_fd, num_conns))
    {
        perror("m_server_socket_init: listen() failed");
        return -1;
    }

    return socket_fd;
}

int m_server_socket_accept(int socket_fd)
{
    struct sockaddr_un peer_addr;
    socklen_t peer_addr_size = sizeof(struct sockaddr_un);
    int cfd = -1;
    if ((cfd = accept(socket_fd, (struct sockaddr*) &peer_addr, &peer_addr_size)) == -1)
    {
        perror("m_server_socket_accept: accept() failed");
        return -1;
    }
    return cfd;
}

int m_client_socket_init(void)
{
    int socket_fd = socket(PF_UNIX, SOCK_STREAM, 0);
    if(socket_fd < 0)
    {
        perror("m_socket_init: socket() failed");
        return -1;
    }
    
    // Attempt to connect
    if (connect(socket_fd,
                (struct sockaddr*) &SOCK_ADDRESS, 
                sizeof(struct sockaddr_un)) == -1)
    {
        perror("m_request_data: connect() failed");
        return -1;
    }

    return socket_fd;
}

int m_receive_int(int socket_fd, int* dest)
{
    char buff[INT_BUFF_LEN] = {0};
    const ssize_t result = recv(socket_fd, buff, sizeof(buff), 0);
    const int scan_res = safe_sscanf(buff, "%i", dest);
    if (scan_res == EOF or scan_res < 1)
    {
        debug_puts("Unable to read valid integer from socket.\n", stderr);
        return -1;
    }
    return result < 1 ? -1 : 0;
}

int m_send_int(int socket_fd, int src)
{
    char buff[INT_BUFF_LEN] = {0};
    if (snprintf(buff, sizeof(buff), "%d", src) < 0)
    {
        debug_puts("Unable to write integer to message buffer.\n", stderr);
        return -1;
    }
    const ssize_t result = send(socket_fd, buff, sizeof(buff), 0);
    return result < 1 ? -1 : 0;
}

ssize_t m_receive_data(int socket_fd, void *dest, const size_t len)
{
    ssize_t num_bytes_received = -1, total_bytes = 0;

    // If request is too large (>4096KB), do muliple reads until completed
    do
    {
        num_bytes_received = recv(socket_fd,
                                  ((unsigned char*) dest) + total_bytes,
                                  len - total_bytes,
                                  0);
        if (num_bytes_received <= 0)
        {
            perror("m_receive_data: UDS read() failed");
            return -1;
        }
        total_bytes += num_bytes_received;
    } while (total_bytes < (ssize_t) len);
    
    return total_bytes;
}

//! Sends data across the socket using the provided connection file descriptor
ssize_t m_send_data(int socket_fd, void const* source, const size_t len)
{
    ssize_t num_bytes_sent = -1, total_bytes = 0;

    // If request is too large (>4096KB), do muliple sends until completed
    do
    {
        num_bytes_sent = send(socket_fd,
                              ((unsigned char*) source) + total_bytes,
                              len - total_bytes,
                              0);
        if (num_bytes_sent <= 0)
        {
            perror("mmq_send_data: UDS write() failed");
            return -1;
        }
        total_bytes += num_bytes_sent;
    } while(total_bytes != (ssize_t) len);

    return total_bytes;
}

int m_close_socket(int socket_fd)
{
    return shutdown(socket_fd, SHUT_RDWR);
}
