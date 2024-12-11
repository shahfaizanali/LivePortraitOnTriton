FROM alfg/nginx-rtmp:latest

# Copy the custom NGINX configuration to the container
COPY nginx.conf /etc/nginx/nginx.conf

RUN mkdir /etc/nginx/ssl

COPY ssl/ /etc/nginx/ssl/

# Expose RTMP port
EXPOSE 1935
EXPOSE 80
EXPOSE 443
# Expose UDP ports for WebRTC
EXPOSE 1024-65535/udp

CMD ["nginx", "-g", "daemon off;"]

