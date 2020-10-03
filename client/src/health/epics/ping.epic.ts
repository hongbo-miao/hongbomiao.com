import { Epic, ofType } from 'redux-observable';
import { Observable, of } from 'rxjs';
import { catchError, map, switchMap } from 'rxjs/operators';
import GraphQLResponse from '../../shared/types/GraphQLResponse.type';
import graphQLSubscriptionClient from '../../shared/utils/graphQLSubscriptionClient';
import HealthActionType from '../actionTypes/Health.actionType';
import HealthAction from '../actions/Health.action';
import GraphQLPing from '../types/GraphQLPing.type';

const subscribePing$ = (query: string): Observable<GraphQLResponse<GraphQLPing>> => {
  return new Observable((observer) => {
    graphQLSubscriptionClient.subscribe(
      {
        query,
      },
      {
        next: (res: GraphQLResponse<GraphQLPing>) => {
          observer.next(res);
        },
        error: (err: Error) => {
          observer.error(err);
        },
        complete: () => {
          observer.complete();
        },
      }
    );
  });
};

const pingEpic: Epic = (action$) =>
  action$.pipe(
    ofType(HealthActionType.SUBSCRIBE_PING),
    switchMap((action) =>
      subscribePing$(action.payload.query).pipe(
        map((res: GraphQLResponse<GraphQLPing>) => HealthAction.receivePingSucceed(res)),
        catchError((err: Error) => of(HealthAction.receivePingFailed(err)))
      )
    )
  );

export default pingEpic;
