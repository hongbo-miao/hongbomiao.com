import { Observable } from 'rxjs';
import { ajax, AjaxResponse } from 'rxjs/ajax';
import config from '../../config';

const graphQLFetch = (query: string): Observable<AjaxResponse> =>
  ajax.post(
    config.graphQLURL,
    {
      query,
    },
    { 'Content-Type': 'application/json' }
  );

export default graphQLFetch;
