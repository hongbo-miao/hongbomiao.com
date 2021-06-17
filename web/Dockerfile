FROM node:16.1.0-alpine AS base
WORKDIR /usr/src/app


FROM base AS builder
COPY ["package.json", "package-lock.json", "./"]
RUN npm ci
COPY . .
# Skip lint check during react-scripts build
RUN rm -f ./.eslintrc.js
RUN npm run build


FROM nginx:1.21.0-alpine AS release
COPY --from=builder /usr/src/app/build /usr/share/nginx/html
COPY conf/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD [ "nginx", "-g", "daemon off;" ]
