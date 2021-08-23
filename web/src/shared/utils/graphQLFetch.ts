import { Observable } from 'rxjs';
import { ajax, AjaxResponse } from 'rxjs/ajax';
import config from '../../config';

const graphQLFetch = (query: string): Observable<AjaxResponse<unknown>> =>
  ajax.post(
    config.graphqlServerGraphQLURL,
    {
      query,
    },
    { 'Content-Type': 'application/json' },
  );

export default graphQLFetch;
