package com.hongbomiao.lance;

import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.FixedSizeListVector;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.flink.table.data.GenericArrayData;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.StringData;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;

/**
 * Converts one row of an Arrow {@link VectorSchemaRoot} produced by a Lance scan into a Flink
 * {@link RowData}, following the field order of the given {@link RowType}.
 *
 * <p>Fields are matched <em>by position</em>, not by name, so the Flink table's column names are
 * free to differ from the Lance dataset's — which matters because a `VECTOR_SEARCH` join puts the
 * search table's columns beside the query table's, and Flink needs those names to be unique.
 *
 * <p>Only the field types the knowledge-base table actually uses are supported (BIGINT, STRING,
 * ARRAY&lt;FLOAT&gt;); this is a learning connector, not a general-purpose Arrow bridge.
 */
public class ConvertArrowRowToRowData {

  private ConvertArrowRowToRowData() {}

  public static RowData convertArrowRowToRowData(
      VectorSchemaRoot vectorSchemaRoot, int rowIndex, RowType rowType) {
    GenericRowData rowData = new GenericRowData(rowType.getFieldCount());
    for (int fieldIndex = 0; fieldIndex < rowType.getFieldCount(); fieldIndex++) {
      LogicalType fieldType = rowType.getTypeAt(fieldIndex);
      FieldVector fieldVector = vectorSchemaRoot.getVector(fieldIndex);
      rowData.setField(fieldIndex, convertArrowValue(fieldVector, rowIndex, fieldType));
    }
    return rowData;
  }

  private static Object convertArrowValue(
      FieldVector fieldVector, int rowIndex, LogicalType fieldType) {
    if (fieldVector.isNull(rowIndex)) {
      return null;
    }
    switch (fieldType.getTypeRoot()) {
      case BIGINT:
        return ((org.apache.arrow.vector.BigIntVector) fieldVector).get(rowIndex);
      case VARCHAR:
        return StringData.fromString(((VarCharVector) fieldVector).getObject(rowIndex).toString());
      case ARRAY:
        // Lance stores vector columns as FixedSizeList, but any other list column comes back as a
        // plain (variable-length) List, so both have to be handled.
        int start;
        int end;
        Float4Vector elements;
        if (fieldVector instanceof FixedSizeListVector) {
          FixedSizeListVector fixedSizeListVector = (FixedSizeListVector) fieldVector;
          start = fixedSizeListVector.getElementStartIndex(rowIndex);
          end = start + fixedSizeListVector.getListSize();
          elements = (Float4Vector) fixedSizeListVector.getDataVector();
        } else {
          ListVector listVector = (ListVector) fieldVector;
          start = listVector.getElementStartIndex(rowIndex);
          end = listVector.getElementEndIndex(rowIndex);
          elements = (Float4Vector) listVector.getDataVector();
        }
        Float[] values = new Float[end - start];
        for (int i = start; i < end; i++) {
          values[i - start] = elements.get(i);
        }
        return new GenericArrayData(values);
      default:
        throw new UnsupportedOperationException(
            "The lance connector does not support field type: " + fieldType);
    }
  }
}
