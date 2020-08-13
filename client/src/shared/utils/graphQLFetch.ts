import { Observable } from 'rxjs';
import { ajax, AjaxResponse } from 'rxjs/ajax';
import Config from '../../Config';

const graphQLFetch = (query: string): Observable<AjaxResponse> =>
  ajax.post(
    Config.graphQLURL,
    {
      query,
    },
    { 'Content-Type': 'application/json' }
  );

export default graphQLFetch;
