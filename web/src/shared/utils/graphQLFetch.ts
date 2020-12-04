import { Observable } from 'rxjs';
import { ajax, AjaxResponse } from 'rxjs/ajax';
import config from '../../config';
import isProduction from './isProduction';

const graphQLFetch = (query: string): Observable<AjaxResponse> =>
  ajax.post(
    isProduction() ? config.prodGraphQLURL : config.devGraphQLURL,
    {
      query,
    },
    { 'Content-Type': 'application/json' }
  );

export default graphQLFetch;
