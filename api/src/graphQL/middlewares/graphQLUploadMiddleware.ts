import { graphqlUploadExpress } from 'graphql-upload';

const graphQLUploadMiddleware = graphqlUploadExpress({
  maxFileSize: 1e6, // 1MB
  maxFiles: 1,
});

export default graphQLUploadMiddleware;
