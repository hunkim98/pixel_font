import { PixelModifyItem } from "dotting";

export const createEmptySquareData = (
  size: number
): Array<Array<PixelModifyItem>> => {
  const data: Array<Array<PixelModifyItem>> = [];
  for (let i = 0; i < size; i++) {
    const row: Array<PixelModifyItem> = [];
    for (let j = 0; j < size; j++) {
      row.push({ rowIndex: i, columnIndex: j, color: "" });
    }
    data.push(row);
  }
  return data;
};
