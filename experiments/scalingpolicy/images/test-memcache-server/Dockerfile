# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM golang:1.27.2 AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY cmd/test-memcache-server-agent/ cmd/test-memcache-server-agent/
RUN CGO_ENABLED=0 go build -o test-memcache-server-agent ./cmd/test-memcache-server-agent/

FROM memcached:1.6
USER root
COPY --from=builder /app/test-memcache-server-agent /usr/local/bin/test-memcache-server-agent
USER memcache
CMD ["test-memcache-server-agent"]
