import {SlSpinner} from '@shoelace-style/shoelace/dist/react';
import {useParams} from 'react-router-dom';
import {EnrichmentType, Field} from '../../fastapi_client';
import {Path, Schema} from '../schema';
import {useGetManifestQuery, useGetMultipleStatsQuery} from '../store/api_dataset';
import {renderPath, renderQuery} from '../utils';
import {Item} from './item_selector';
import {getLeafsByEnrichmentType} from './search_box_utils';

export function ColumnSelector({
  onSelect,
  leafFilter,
  enrichmentType,
}: {
  onSelect: (path: Path, field: Field) => void;
  leafFilter?: (leaf: [Path, Field], embeddings: string[]) => boolean;
  enrichmentType?: EnrichmentType;
}) {
  const {namespace, datasetName} = useParams<{namespace: string; datasetName: string}>();
  if (namespace == null || datasetName == null) {
    throw new Error('Invalid route');
  }

  const query = useGetManifestQuery({
    namespace,
    datasetName,
  });

  const dataSchema = query.currentData?.dataset_manifest.data_schema;
  const schema = dataSchema != null ? new Schema(dataSchema) : null;
  const pathToEmbeddings: {[path: string]: string[]} = {};
  for (const {column, embedding} of query.currentData?.dataset_manifest.embedding_manifest
    .indexes || []) {
    if (embedding.embedding_name == null) {
      continue;
    }
    const col = renderPath(column as Path);
    if (pathToEmbeddings[col] == null) {
      pathToEmbeddings[col] = [];
    }
    pathToEmbeddings[col].push(embedding.embedding_name);
  }
  const leafs = schema != null ? getLeafsByEnrichmentType(schema.leafs, enrichmentType) : null;

  const inFilterLeafs: [Path, Field][] = [];
  const outFilterLeafs: [Path, Field][] = [];
  if (leafFilter != null) {
    for (const leaf of leafs || []) {
      if (leafFilter(leaf, pathToEmbeddings[renderPath(leaf[0])])) {
        inFilterLeafs.push(leaf);
      } else {
        outFilterLeafs.push(leaf);
      }
    }
  } else {
    inFilterLeafs.push(...(leafs || []));
  }

  const stats = useGetMultipleStatsQuery(
    {namespace, datasetName, leafPaths: leafs?.map(([path]) => path) || []},
    {skip: schema == null}
  );

  const showAvgLength = enrichmentType == 'text';

  return renderQuery(query, () => {
    return (
      <>
        <div
          className="mb-1 flex w-full justify-between
                     border-b-2 border-gray-100 px-4 pb-1 text-sm font-medium"
        >
          <div className="truncate">column</div>
          <div className="flex flex-row items-end justify-items-end text-end">
            <div className="w-24 truncate">count</div>
            {showAvgLength ? <div className="w-24 truncate">avg length</div> : <></>}
            <div className="w-24 truncate">dtype</div>
          </div>
        </div>
        {inFilterLeafs!.map(([path, field], i) => {
          const totalCount = stats?.currentData?.[i].total_count;
          const avgTextLength = stats?.currentData?.[i].avg_text_length;
          const avgTextLengthDisplay =
            avgTextLength != null ? Math.round(avgTextLength).toLocaleString() : null;
          const renderedPath = renderPath(path);
          return (
            <Item key={i} onSelect={() => onSelect(path, field)}>
              <div className="flex w-full justify-between">
                <div className="truncate">{renderedPath}</div>
                <div className="flex flex-row items-end justify-items-end text-end">
                  <div className="w-24 truncate">
                    {totalCount == null ? <SlSpinner></SlSpinner> : totalCount.toLocaleString()}
                  </div>
                  {showAvgLength ? (
                    <div className="w-24 truncate">
                      {avgTextLength == null ? <SlSpinner></SlSpinner> : avgTextLengthDisplay}
                    </div>
                  ) : (
                    <></>
                  )}
                  <div className="w-24 truncate">{field.dtype}</div>
                </div>
              </div>
            </Item>
          );
        })}
        {outFilterLeafs!.map(([path, field], i) => {
          const totalCount = stats?.currentData?.[i].total_count;
          const avgTextLength = stats?.currentData?.[i].avg_text_length;
          const avgTextLengthDisplay =
            avgTextLength != null ? Math.round(avgTextLength).toLocaleString() : null;
          const renderedPath = renderPath(path);
          return (
            <Item key={i} onSelect={() => onSelect(path, field)} disabled={true}>
              <div className="flex w-full justify-between opacity-50">
                <div className="truncate">{renderedPath}</div>
                <div className="flex flex-row items-end justify-items-end text-end">
                  <div className="w-24 truncate">
                    {totalCount == null ? <SlSpinner></SlSpinner> : totalCount.toLocaleString()}
                  </div>
                  <div className="w-24 truncate">
                    {avgTextLength == null ? <SlSpinner></SlSpinner> : avgTextLengthDisplay}
                  </div>
                  <div className="w-24 truncate">{field.dtype}</div>
                </div>
              </div>
            </Item>
          );
        })}
      </>
    );
  });
}