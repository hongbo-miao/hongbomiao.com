<p align="center">
  <a href="https://www.hongbomiao.com"><img width="100" src="https://user-images.githubusercontent.com/3375461/93688946-821f1880-fafc-11ea-8918-374f21f4ac6e.png" alt="Flying" /></a>
</p>

<h2 align="center">
  HONGBO MIAO
</h2>

<p align="center">
  Making magic happen
</p>

<br />

<p align="center">
  <a href="https://github.com/Hongbo-Miao/hongbomiao.com/actions"><img alt="GitHub Actions" src="https://img.shields.io/github/workflow/status/Hongbo-Miao/hongbomiao.com/Deploy" /></a>
  <a href="https://app.fossa.io/projects/git%2Bgithub.com%2FHongbo-Miao%2Fhongbomiao.com"><img alt="FOSSA Status" src="https://app.fossa.io/api/projects/git%2Bgithub.com%2FHongbo-Miao%2Fhongbomiao.com.svg?type=shield" /></a>
  <a href="https://depfu.com/github/Hongbo-Miao/hongbomiao.com?project_id=29781"><img alt="Depfu" src="https://badges.depfu.com/badges/e337a1462a48dbcb803c98b5b2157aa7/overview.svg" /></a>
  <a href="https://stats.uptimerobot.com/RoOoGTvyWN"><img alt="Uptime Robot status" src="https://img.shields.io/uptimerobot/status/m783305207-c7842815153e530df85633fe" /></a>
  <a href="https://www.http3check.net/?host=hongbomiao.com"><img alt="HTTP/3" src="https://img.shields.io/badge/http%2F3-supported-brightgreen" /></a>
  <a href="https://github.com/commitizen/cz-cli"><img alt="Commitizen friendly" src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg" /></a>
</p>

<p align="center">
  <a href="https://goreportcard.com/report/github.com/Hongbo-Miao/hongbomiao.com"><img alt="Go Report Card" src="https://goreportcard.com/badge/github.com/Hongbo-Miao/hongbomiao.com" /></a>
  <a href="https://observatory.mozilla.org/analyze/www.hongbomiao.com"><img alt="Mozilla Observatory grade" src="https://img.shields.io/mozilla-observatory/grade/www.hongbomiao.com" /></a>
  <a href="https://app.codacy.com/app/Hongbo-Miao/hongbomiao.com"><img alt="Codacy grade" src="https://img.shields.io/codacy/grade/dc922acc14014b4abc978afd0810e56b" /></a>
  <a href="https://codeclimate.com/github/Hongbo-Miao/hongbomiao.com/maintainability"><img alt="Code Climate maintainability" src="https://img.shields.io/codeclimate/maintainability/Hongbo-Miao/hongbomiao.com" /></a>
  <a href="https://codeclimate.com/github/Hongbo-Miao/hongbomiao.com/trends/technical_debt"><img alt="Code Climate technical debt" src="https://img.shields.io/codeclimate/tech-debt/Hongbo-Miao/hongbomiao.com" /></a>
</p>

<p align="center">
  <a href="https://hstspreload.org/?domain=www.hongbomiao.com"><img alt="Chromium HSTS preload" src="https://img.shields.io/hsts/preload/www.hongbomiao.com" /></a>
  <a href="https://codecov.io/gh/Hongbo-Miao/hongbomiao.com"><img alt="Codecov" src="https://img.shields.io/codecov/c/github/Hongbo-Miao/hongbomiao.com" /></a>
  <a href="https://github.com/prettier/prettier"><img alt="Code style" src="https://img.shields.io/badge/code_style-prettier-ff69b4.svg" /></a>
  <a href="https://github.com/Hongbo-Miao/hongbomiao.com/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Hongbo-Miao/hongbomiao.com" /></a>
</p>

<p align="center">
  <a href="https://github.com/Hongbo-Miao/hongbomiao.com"><img alt="Lines of code" src="https://img.shields.io/tokei/lines/github/Hongbo-Miao/hongbomiao.com" /></a>
  <a href="https://github.com/Hongbo-Miao/hongbomiao.com"><img alt="Code size" src="https://img.shields.io/github/languages/code-size/Hongbo-Miao/hongbomiao.com" /></a>
  <a href="https://github.com/Hongbo-Miao/hongbomiao.com/graphs/contributors"><img alt="Commit activity" src="https://img.shields.io/github/commit-activity/y/Hongbo-Miao/hongbomiao.com"></a>
</p>

<p align="center">
  <a href="https://twitter.com/hongbo_miao"><img alt="Twitter follow" src="https://img.shields.io/twitter/follow/hongbo_miao?label=Follow&style=social" /></a>
  <a href="https://github.com/hongbo-miao"><img alt="GitHub follow" src="https://img.shields.io/github/followers/Hongbo-Miao?label=Follow&style=social" /></a>
</p>

<p align="center">
  <a href="https://www.chromatic.com/builds?appId=5d626a63a601530020759b10"><img alt="Storybook" src="https://cdn.jsdelivr.net/gh/storybookjs/brand@master/badge/badge-storybook.svg" /></a>
</p>

<p align="center">
  <a href="https://www.hongbomiao.com">hongbomiao.com</a>
</p>

---

## Introduction

Personal cutting-edge technology lab.

[https://www.hongbomiao.com](https://www.hongbomiao.com)

[![hongbomiao.com record](https://user-images.githubusercontent.com/3375461/100162554-deccf400-2eee-11eb-89b4-49cd6c23026d.gif)](https://www.hongbomiao.com)

## Architecture

This diagram shows the architecture of this repository.

It is way over-engineering. Please make sure to know the tradeoffs before onboarding any technology to your project.

![Architecture](https://user-images.githubusercontent.com/3375461/168464738-2e49f1e0-8b8f-42e0-a7e0-8e69dd7e0eab.png)

## Setup

```shell
make setup
```

### Clean

```shell
make clean
```

## Tech Stack

### Web

- **React** - Web framework
- **Redux** - State container
- **React Query** - Hooks for fetching, caching and updating asynchronous data
- **redux-observable** - Side effects
- **RxJS** - Asynchronous programming with observable streams
- **graphql-tag** - GraphQL query parsing
- **Bulma** - CSS framework
- **PurgeCSS** - Unused CSS removing
- **Jest** - Unit testing, snapshot Testing
- **React Testing Library** - React component testing
- **Storybook** - Visual testing
- **rxjs/testing** - Marble testing
- **Cypress** - End-to-end testing
- **Lighthouse CI** - Performance, accessibility, SEO, progressive web app (PWA) analysis

### Mobile

- **React Native** - Mobile application framework
- **UI Kitten** - UI library
- **React Native Testing Library** - React Native component testing

### API Server - Go

- **Gin** - Web Framework
- **gRPC** - Remote procedure call (RPC) framework
- **graphql-go** - GraphQL
- **jwt-go** - JWT
- **gin-contrib/cors** - CORS
- **opa** - Open Policy Agent
- **dgo** - Dgraph client
- **minio-go** - MinIO client
- **go-redis** - Redis client
- **pgx** - PostgreSQL driver
- **Resty** - HTTP client
- **Squirrel** - SQL query builder
- **apm-agent-go** - APM agent
- **OpenTelemetry Go** - OpenTelemetry
- **Prometheus Go** - Prometheus
- **Testify** - Unit testing
- **GoDotEnv** - Environment variables loading
- **jsonparser** - JSON parser
- **zerolog** - Logging

### API Server - Node.js

- **Express** - Node.js web application framework
- **GraphQL.js**, **express-graphql** - GraphQL
  - **graphql-ws**, **graphql-subscriptions** - GraphQL subscriptions
  - **graphql-upload** - GraphQL upload
  - **graphql-shield** - GraphQL permissions
  - **graphql-depth-limit** - GraphQL depth limit
  - **graphql-query-complexity** - GraphQL query complexity analysis
- **DataLoader** - Batching and caching
- **Knex.js** - SQL query builder
- **node-postgres** - PostgreSQL client
- **ioredis** - Redis client
- **rate-limiter-flexible** - Rate limiting
- **expressjs/cors** - CORS
- **csurf** - CSRF protection
- **jsonwebtoken**, **express-jwt** - JSON Web Tokens (JWT)
- **bcrypt** - Password hashing
- **axios** - HTTP client
- **Helmet** - HTTP header `Content-Security-Policy`, `Expect-CT`, `Referrer-Policy`, `Strict-Transport-Security`, `X-Content-Type-Options`, `X-DNS-Prefetch-Control`, `X-Download-Options`, `X-Frame-Options`, `X-Permitted-Cross-Domain-Policies`, `X-XSS-Protection`
- **Report To** - HTTP header `Report-To`
- **Network Error Logging** - HTTP header `NEL`
- **express-request-id** - HTTP header `X-Request-ID`
- **response-time** - HTTP header `X-Response-Time`
- **connect-timeout** - Request timeout
- **Terminus** - Health check and graceful shutdown
- **Opossum** - Circuit breaker
- **pino** - Logging
- **dotenv-flow** - Environment variables loading
- **Stryker** - Mutation testing
- **SuperTest** - HTTP testing
- **autocannon** - HTTP benchmarking
- **Clinic.js** - Performance profiling

### Data

- **Trino** - Distributed SQL query engine
- **YugabyteDB** - Distributed SQL database
- **TimescaleDB** - Time-series SQL database
- **Dgraph** - Distributed graph database
- **Elasticsearch** - Distributed document-oriented search engine
- **PostgreSQL** - Object-relational database
- **KeyDB** - High performance fork of Redis
- **MinIO** - High performance object storage
- **Flink** - Stream processing framework
  - **flink-streaming-java** - Flink
  - **flink-connector-twitter** - Flink Twitter connector
  - **flink-connector-jdbc** - Flink JDBC Connector
  - **flink-connector-redis** - Flink Redis connector
- **Kafka** - Distributed event streaming platform
  - **Debezium** - Distributed change-data-capture platform
  - **debezium-connector-postgres** - PostgreSQL connector
  - **kafka-connect-elasticsearchkafka-connect-elasticsearch** - Elasticsearch sink connector
  - **http-connector-for-apache-kafka** - HTTP sink connector
- **Superset** - Data exploration and data visualization platform
- **golang-migrate/migrate** - Database migrations

### Cloud Native

- **Hasura** - GraphQL Engine
  - **hasura-metric-adapter** - Hasura GraphQL Engine metric adapter
- **Ory Hydra** - OAuth 2.0 and OpenID Connect server
- **Terraform** - Infrastructure as code
- **TorchServe** - PyTorch models serving
- **Linkerd** - Service mesh
- **Traefik** - HTTP reverse proxy and load balancer
- **Open Policy Agent (OPA)** - Policy-based control
- **OPAL** - Open-policy administration layer
- **Kibana** - Data visualization dashboard for Elasticsearch
- **Elastic APM** - Application performance monitoring
- **OpenTelemetry** - Observability framework
- **Jaeger** - Distributed tracing system
- **Grafana** - Monitoring and observability platform
- **Prometheus** - Monitoring system
- **Thanos** - Highly available Prometheus setup with long term storage capabilities
- **Fluent Bit** - Log processor and forwarder
- **Pixie** - Observability tool for Kubernetes applications
- **Docker** - Container
- **Skaffold** - Continuous development for Kubernetes applications
- **Multipass** - VM manager
- **Locust** - Load testing
- **NGINX** - Reverse proxy, load balancer
- **Cloudflare Tunnel** - Tunneling
- **Kubernetes** - Container-orchestration system
- **K3s** - Lightweight Kubernetes

### Ops

- **Argo CD** - Declarative GitOps CD for Kubernetes
- **Rancher** - Kubernetes container management platform
- **Goldilocks** - Kubernetes resource requests recommending
- **Polaris** - Kubernetes best practices validating
- **Kubecost** - Kubernetes cost monitoring and management
- **Sloop** - Kubernetes history visualization
- **Ansible** - IT automation system
- **CodeQL** - Variant analysis

### Neural Network

- **PyTorch** - Machine learning framework
  - **PyTorch Geometric** - PyTorch geometric deep learning extension
- **OGB** - Open graph benchmark
- **Rasa** - Machine learning framework for automated text and voice-based conversations
- **CML** - Continuous machine learning
- **DVC** - Data version control
- **[Weights & Biases](https://wandb.ai/hongbo-miao/graph-neural-network)** - Machine learning experiment tracking

### Quantum Computing

- **Qiskit** - Quantum computing SDK

### OPAL Client

- **asyncpg** - PostgreSQL client
- **pydantic** - Data validation
- **Tenacity** - General-purpose retrying library

### Ethereum

- **Solidity** - Contract-oriented programming language
- **solc-js** - JavaScript bindings for the Solidity compiler

### Code

- **Prettier** - Code formatter
- **gofmt** - Go code formatter
- **opa** - Rego code formatter
- **Black** - Python code formatter
- **tsc** - TypeScript static type checker
- **Mypy** - Python static type checker
- **ESLint** - JavaScript linter
- **Stylelint** - CSS linter
- **golangci-lint** - Go linter
- **Buf** - Protocol Buffers linter
- **solhint** - Solidity linter
- **markdownlint-cli2** - Markdown linter
- **ShellCheck** - Shell linter
- **hadolint** - Dockerfile linter
- **Kubeval** - Kubernetes configuration file linter
- **commitlint** - Commit message linter
- **Husky** - Bad git commit and push preventing

### Services

- **Sentry** - Error tracking
- **Report URI** - Security reporting
- **Google Tag Manager** - Tag management
- **Google Analytics** - Web analytics
- **FullStory** - Experience analytics, session replay, heatmaps
- **Namecheap** - Domain
- **Cloudflare** - CDN, DNS, DDoS protection
- **Discord** - ChatOps
- **Opsgenie** - Incident management platform
- **[GitHub Actions](https://github.com/Hongbo-Miao/hongbomiao.com/actions)** - Continuous integration
- **[SonarCloud](https://sonarcloud.io/dashboard?id=Hongbo-Miao_hongbomiao.com)**, **[Codacy](https://app.codacy.com/app/Hongbo-Miao/hongbomiao.com)**, **[Code Climate](https://codeclimate.com/github/Hongbo-Miao/hongbomiao.com/maintainability)**, **[LGTM](https://lgtm.com/projects/g/Hongbo-Miao/hongbomiao.com)** - Code reviews and analytics
- **[Codecov](https://codecov.io/gh/Hongbo-Miao/hongbomiao.com)** - Code coverage reports
- **[Chromatic](https://www.chromatic.com/builds?appId=5d626a63a601530020759b10)** - UI reviewing and feedback collecting
- **[HTTP/3 Check](https://www.http3check.net/?host=www.hongbomiao.com)** - HTTP/3 checking
- **[hstspreload.org](https://hstspreload.org/?domain=hongbomiao.com)** - HSTS checking
- **[Mozilla Observatory](https://observatory.mozilla.org/analyze/www.hongbomiao.com)**, **[Snyk](https://snyk.io/test/github/Hongbo-Miao/hongbomiao.com)** - Security monitoring
- **[Depfu](https://depfu.com/github/Hongbo-Miao/hongbomiao.com?project_id=23463)**, **[Requires.io](https://requires.io/github/Hongbo-Miao/hongbomiao.com/requirements)** - Dependency monitoring
- **[Uptime Robot](https://stats.uptimerobot.com/RoOoGTvyWN)** - Uptime monitoring
- **[FOSSA](https://app.fossa.io/projects/git%2Bgithub.com%2FHongbo-Miao%2Fhongbomiao.com)** - License compliance

### Bots

- **Renovate** - Dependency update
- **Mergify** - Automatically merging
- **Stale** - Stale issues and pull requests closing
- **ImgBot** - Image compression
- **semantic-release** - Version management and package publishing

## Highlights

### Cloud Native

#### Pixie - Kubernetes Application Observing

![Pixie screenshot](https://user-images.githubusercontent.com/3375461/168404534-18960440-3d91-4b03-9775-321364c3fcf8.jpg)

![Pixie screenshot](https://user-images.githubusercontent.com/3375461/168404530-cdc5fa7b-b9ec-4ce3-b573-5312badf2e7b.jpg)

![Pixie screenshot](https://user-images.githubusercontent.com/3375461/168404538-98a197d9-632e-4252-b7e5-d1b94961c3d1.jpg)

![Pixie screenshot](https://user-images.githubusercontent.com/3375461/168404542-2d43d21c-b6f4-4af6-93b7-b5e76c104268.jpg)

#### Linkerd - Service Mesh

![Linkerd screenshot](https://user-images.githubusercontent.com/3375461/127684629-a0d9d76b-cbc6-465f-80ea-10c3e06f7eac.png)

#### Hasura - GraphQL Engine

![Hasura GraphQL Engine screenshot](https://user-images.githubusercontent.com/3375461/167373764-3fd97b12-034c-42bd-84d5-c65ecd068ae1.jpg)

#### Traefik - Reverse Proxy and Load Balancer

![Traefik screenshot](https://user-images.githubusercontent.com/3375461/168451816-fee9aa54-d4b7-430d-bca3-a8f453931a35.jpg)

#### Elastic APM - Application Performance Management

![Elastic APM screenshot](https://user-images.githubusercontent.com/3375461/128647400-7377f888-6c76-4b13-8bce-50ad7afdb3c3.png)

#### Jaeger - Distributed Tracing

![Jaeger screenshot](https://user-images.githubusercontent.com/3375461/90900854-9e943c00-e3fc-11ea-9628-682a605972eb.jpg)

#### Grafana - Monitoring and Observability Platform

![Grafana screenshot](https://user-images.githubusercontent.com/3375461/163708604-89ce4617-8fb7-463f-86a0-11a3c5c73bd9.png)

#### Prometheus - Metrics

![Prometheus screenshot](https://user-images.githubusercontent.com/3375461/90955864-d14d3b80-e4b3-11ea-926b-8012cadb4f70.jpg)

#### Kibana

![Kibana screenshot](https://user-images.githubusercontent.com/3375461/90955224-50d80c00-e4ae-11ea-9345-dfa8e97ed41a.jpg)

#### Locust - Load Testing

![Locust screenshot](https://user-images.githubusercontent.com/3375461/98866512-0613d200-24a8-11eb-8275-d245efdc4727.jpg)

### Ops

#### Argo CD - GitOps

![Argo CD screenshot](https://user-images.githubusercontent.com/3375461/127684622-28c23303-1b93-476d-9080-6194471a8c9a.png)

#### Discord - ChatOps

![Discord screenshot](https://user-images.githubusercontent.com/3375461/135687134-aaad261c-dee9-4a70-b8b2-2da393e114cb.png)

#### Rancher - Kubernetes Container Management

![Rancher screenshot](https://user-images.githubusercontent.com/3375461/168413513-fb747e9f-ac75-4920-aa9e-f9253b8f724f.jpg)

#### Kubecost - Kubernetes Cost Monitoring

![Kubecost screenshot](https://user-images.githubusercontent.com/3375461/167351502-ee32bb31-3499-4a9c-9dcc-87b38099aa62.jpg)

#### Polaris - Kubernetes Best Practices Validating

![Polaris screenshot](https://user-images.githubusercontent.com/3375461/167352130-75b7c8ee-d7e1-4731-9765-c4d05e22f684.jpg)

#### Goldilocks - Kubernetes Resource Requests Recommending

![Goldilocks screenshot](https://user-images.githubusercontent.com/3375461/167352330-f2d99896-e99d-4e89-876c-91ea0741e1c2.jpg)

#### Sloop - Kubernetes History Visualization

![Sloop screenshot](https://user-images.githubusercontent.com/3375461/167351205-0f7e0921-07ff-4072-b5b9-c343d88f25c4.jpg)

### Data

#### Flink - Stream Processing

![Flink screenshot](https://user-images.githubusercontent.com/3375461/129500704-9032b559-5dc5-4385-99eb-3f7d4a1f1086.png)

#### Dgraph - Distributed Graph Database

![Dgraph screenshot](https://user-images.githubusercontent.com/3375461/126815004-925b6be5-6e44-44be-9193-46b018ec4bf7.png)

#### Redis with RedisGraph Module

![Redis screenshot](https://user-images.githubusercontent.com/3375461/167368266-8f8d27b1-ec58-48c5-b6bd-030dc2970cc8.jpg)

### Machine Learning

#### Contextual AI assistant

Chatbot on Telegram powered by Rasa.

![Telegram screenshot](https://user-images.githubusercontent.com/3375461/148026649-70dca296-7813-4d58-bc82-d7a2c5b8576e.png)

#### Weights & Biases - Machine Learning Experiment Tracking

[Distributed hyperparameter optimization result](https://wandb.ai/hongbo-miao/graph-neural-network/sweeps/p0fgtvcf) by Weights & Biases.

[![Weights & Biases screenshot](https://user-images.githubusercontent.com/3375461/112082183-dbf1bf80-8bbf-11eb-9c81-675cc0bd2763.png)](https://wandb.ai/hongbo-miao/graph-neural-network/sweeps/p0fgtvcf)

### HTTP/3

The website [supports HTTP/3](https://www.http3check.net/?host=hongbomiao.com).

[![HTTP/3 screenshot](https://user-images.githubusercontent.com/3375461/92599407-cdefe780-f2dc-11ea-8bf9-dc153187287f.jpg)](https://www.http3check.net/?host=hongbomiao.com)

### AVIF

Images on the website are using AVIF format.

> “Roughly speaking, at an acceptable quality, the WebP is almost half the size of JPEG, and AVIF is under half the size of WebP.” – [Jake Archibald, 2020](https://jakearchibald.com/2020/avif-has-landed/)

### Security

Below is the website [security report](https://observatory.mozilla.org/analyze/www.hongbomiao.com) generated by Mozilla Observatory.

[![Mozilla Observatory screenshot](https://user-images.githubusercontent.com/3375461/148025144-57d3c888-7ddd-4242-90ab-9759642f393d.png)](https://observatory.mozilla.org/analyze/www.hongbomiao.com)

### Profiling

Profiling result by Clinic.js and autocannon.

![Profiling screenshot](https://user-images.githubusercontent.com/3375461/94975997-100bf200-0546-11eb-9284-db40711a3052.jpg)

### Automation

This [pull request](https://github.com/Hongbo-Miao/hongbomiao.com/pull/342) shows how these technologies work together from different aspects to achieve automation.

[![Automation screenshot](https://user-images.githubusercontent.com/3375461/91974789-e5fdbf00-ed50-11ea-8540-5d429312d053.jpg)](https://github.com/Hongbo-Miao/hongbomiao.com/pull/342)
