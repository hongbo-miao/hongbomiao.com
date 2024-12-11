import { useQuery } from '@tanstack/react-query';
import { Observable } from 'rxjs';
import GraphQLResponse from '../../shared/types/GraphQLResponse';
import graphQLSubscriptionClient from '../../shared/utils/graphQLSubscriptionClient';
import pingSubscription from '../queries/pingSubscription';
import GraphQLPing from '../types/GraphQLPing';

const subscribePing$ = (query: string): Observable<GraphQLResponse<GraphQLPing>> => {
  return new Observable((observer) => {
    return graphQLSubscriptionClient.subscribe(
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
      },
    );
  });
};

const useHealthSubscription = () => {
  useQuery({
    queryKey: ['health', 'ping'],
    queryFn: () => {
      const subscription = subscribePing$(pingSubscription);
      return new Promise((resolve) => {
        subscription.subscribe({
          next: (response) => {
            resolve(response);
          },
          error: (error) => {
            console.error('Health subscription error:', error);
          },
        });
      });
    },
    refetchInterval: 30000, // Refetch every 30 seconds
  });
};

export default useHealthSubscription;
