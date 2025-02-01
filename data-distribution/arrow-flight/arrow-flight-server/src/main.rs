use arrow::array::RecordBatchReader;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::flight_descriptor::DescriptorType;
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, Location, PollInfo, PutResult, SchemaResult, Ticket,
};
use futures_util::future::{self, Ready};
use futures_util::{stream, TryStreamExt};
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tonic::codegen::Bytes;
use tonic::{transport::Server, Request, Response, Status, Streaming};

struct FlightSqlServer {
    parquet_files: HashMap<String, PathBuf>,
    location: String,
}

impl FlightSqlServer {
    async fn new<P: AsRef<Path>>(data_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut parquet_files = HashMap::new();
        let mut entries = fs::read_dir(data_path.as_ref()).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("parquet") {
                let file_name = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap()
                    .to_string();
                parquet_files.insert(file_name, path);
            }
        }
        Ok(Self {
            parquet_files,
            location: "grpc://[::1]:50841".to_string(),
        })
    }

    fn make_flight_info(&self, file_name: &str) -> Result<FlightInfo, Status> {
        let path = self
            .parquet_files
            .get(file_name)
            .ok_or_else(|| Status::not_found(format!("File not found: {}", file_name)))?;

        let file = std::fs::File::open(path)
            .map_err(|e| Status::internal(format!("Failed to open file: {}", e)))?;
        let reader = SerializedFileReader::new(file)
            .map_err(|e| Status::internal(format!("Failed to create Parquet reader: {}", e)))?;

        let metadata = reader.metadata();
        let file = std::fs::File::open(path)
            .map_err(|e| Status::internal(format!("Failed to open file: {}", e)))?;
        let arrow_reader = ParquetRecordBatchReader::try_new(file, 1000)
            .map_err(|e| Status::internal(format!("Failed to create Arrow reader: {}", e)))?;
        let arrow_schema = arrow_reader.schema();

        let descriptor = FlightDescriptor {
            r#type: DescriptorType::Path as i32,
            cmd: Bytes::new(),
            path: vec![file_name.to_string()],
        };

        let endpoint = arrow_flight::FlightEndpoint {
            ticket: Some(Ticket {
                ticket: Bytes::from(file_name.as_bytes().to_vec()),
            }),
            location: vec![Location {
                uri: self.location.clone(),
            }],
            app_metadata: Bytes::new(),
            expiration_time: None,
        };

        let mut schema_buf = Vec::new();
        {
            let mut writer =
                arrow::ipc::writer::StreamWriter::try_new(&mut schema_buf, &arrow_schema).map_err(
                    |e| Status::internal(format!("Failed to create schema writer: {}", e)),
                )?;
            writer
                .finish()
                .map_err(|e| Status::internal(format!("Failed to finish schema writer: {}", e)))?;
        }

        Ok(FlightInfo {
            schema: Bytes::from(schema_buf),
            flight_descriptor: Some(descriptor),
            endpoint: vec![endpoint],
            total_records: {
                let mut total = 0;
                for i in 0..metadata.num_row_groups() {
                    total += metadata.row_group(i).num_rows();
                }
                total
            },
            total_bytes: {
                let mut total = 0;
                for i in 0..metadata.num_row_groups() {
                    total += metadata.row_group(i).compressed_size();
                }
                total
            },
            ordered: false,
            app_metadata: Bytes::new(),
        })
    }
}

#[tonic::async_trait]
impl FlightService for FlightSqlServer {
    type HandshakeStream = stream::Once<Ready<Result<HandshakeResponse, Status>>>;
    type ListFlightsStream = stream::BoxStream<'static, Result<FlightInfo, Status>>;
    type DoGetStream = stream::BoxStream<'static, Result<FlightData, Status>>;
    type DoPutStream = stream::BoxStream<'static, Result<PutResult, Status>>;
    type DoActionStream = stream::BoxStream<'static, Result<arrow_flight::Result, Status>>;
    type ListActionsStream = stream::BoxStream<'static, Result<ActionType, Status>>;
    type DoExchangeStream = stream::BoxStream<'static, Result<FlightData, Status>>;

    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        let response = HandshakeResponse::default();
        let result = stream::once(future::ready(Ok(response)));
        Ok(Response::new(result))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        let flights = self
            .parquet_files
            .keys()
            .map(|file_name| self.make_flight_info(file_name))
            .collect::<Vec<_>>();
        let stream = Box::pin(stream::iter(flights)) as Self::ListFlightsStream;
        Ok(Response::new(stream))
    }

    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let descriptor = request.into_inner();
        if descriptor.path.is_empty() {
            return Err(Status::invalid_argument("No path specified"));
        }
        let file_name = &descriptor.path[0];
        let info = self.make_flight_info(file_name)?;
        Ok(Response::new(info))
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = request.into_inner();
        let file_name = String::from_utf8(ticket.ticket.to_vec())
            .map_err(|_| Status::invalid_argument("Invalid ticket"))?;
        let path = self
            .parquet_files
            .get(&file_name)
            .ok_or_else(|| Status::not_found(format!("File not found: {}", file_name)))?;

        let file = std::fs::File::open(path)
            .map_err(|e| Status::internal(format!("Failed to open file: {}", e)))?;
        let arrow_reader = ParquetRecordBatchReader::try_new(file, 1000)
            .map_err(|e| Status::internal(format!("Failed to create Arrow reader: {}", e)))?;

        let schema = arrow_reader.schema();
        let stream = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(stream::iter(arrow_reader.map(|batch| {
                batch.map_err(|e| FlightError::from_external_error(Box::new(e)))
            })))
            .map_err(|e| Status::internal(format!("Failed to encode data: {}", e)));

        Ok(Response::new(Box::pin(stream.map_err(|e| {
            Status::internal(format!("Failed to stream data: {}", e))
        }))))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info not implemented"))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented("get_schema not implemented"))
    }

    async fn do_put(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented("do_put not implemented"))
    }

    async fn do_action(
        &self,
        _request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        Err(Status::unimplemented("do_action not implemented"))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        Err(Status::unimplemented("list_actions not implemented"))
    }

    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("do_exchange not implemented"))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50841".parse()?;
    let server = FlightSqlServer::new("data").await?;
    println!("Flight SQL Server listening on {}", addr);
    Server::builder()
        .add_service(FlightServiceServer::new(server))
        .serve(addr)
        .await?;
    Ok(())
}
