use crate::graphql::schema::ApiSchema;

#[derive(Clone)]
pub struct ApplicationState {
    pub schema: ApiSchema,
}
