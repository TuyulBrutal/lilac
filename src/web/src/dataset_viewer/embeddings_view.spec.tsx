import {screen} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {SpyInstance, vi} from 'vitest';
import {CancelablePromise, DatasetsService, WebManifest} from '../../fastapi_client';
import {renderWithProviders} from '../../tests/utils';
import {EmbedingsView} from './embeddings_view';

describe('EmbeddingsView', () => {
  let spy: SpyInstance<
    [namespace: string, datasetName: string],
    CancelablePromise<Partial<WebManifest>>
  >;

  const embeddingsView = <EmbedingsView namespace="test-namespace" datasetName="test-dataset" />;

  beforeEach(() => {
    spy = vi.spyOn(DatasetsService, 'getManifest');
  });

  afterEach(() => {
    spy.mockRestore();
  });

  it('should call the api with correct parameters', async () => {
    spy.mockResolvedValueOnce({});

    renderWithProviders(embeddingsView);

    expect(spy).toBeCalledWith('test-namespace', 'test-dataset');
  });

  it('renders when no embeddings are created', async () => {
    spy.mockResolvedValueOnce({});

    renderWithProviders(embeddingsView);

    expect(await screen.findByText('No embeddings computed')).toBeInTheDocument();
  });

  it('renders when embeddings are created', async () => {
    spy.mockResolvedValueOnce({
      dataset_manifest: {
        namespace: 'namespace',
        dataset_name: 'dataset_name',
        data_schema: {
          fields: {},
        },
        embedding_manifest: {
          indexes: [
            {
              column: ['column_name'],
              embedding: {
                embedding_name: 'cohere',
              },
            },
          ],
        },
        entity_indexes: [],
        num_items: 1,
      },
    });

    renderWithProviders(embeddingsView);

    const rows = await screen.findAllByRole('row');
    expect(rows).toHaveLength(2);
    expect(rows[1]).toHaveTextContent('column_name');
    expect(rows[1]).toHaveTextContent('cohere');
  });

  test('clicking add embedding button opens search box', async () => {
    spy.mockResolvedValueOnce({});
    const {store} = renderWithProviders(embeddingsView);

    expect(store.getState().app.searchBoxOpen).toBe(false);

    await userEvent.click(await screen.findByText('Add Embedding'));

    expect(store.getState().app.searchBoxOpen).toBe(true);
    expect(store.getState().app.searchBoxPages).toEqual([
      {type: 'compute-embedding-index', name: 'compute embeddings'},
    ]);
  });
});