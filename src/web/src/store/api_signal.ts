/**
 * RTK Query APIs for the dataset service: 'dataset' tag in FastAPI.
 */
import {createApi} from '@reduxjs/toolkit/dist/query/react';
import {SignalInfo, SignalsService} from '../../fastapi_client';
import {fastAPIBaseQuery} from './api_utils';

const SIGNALS_TAG = 'signals';
export const signalApi = createApi({
  reducerPath: 'signalApi',
  baseQuery: fastAPIBaseQuery(),
  tagTypes: [SIGNALS_TAG],
  endpoints: (builder) => ({
    getSignals: builder.query<SignalInfo[], void>({
      query: () => () => SignalsService.getSignals(),
    }),
  }),
});

export const {useGetSignalsQuery} = signalApi;