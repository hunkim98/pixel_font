"use client";

import ProbabilityChart from "@/components/Chart/ProbabilityChart";
import { hexToRgb, rgbToGrayscale } from "@/utils/color";
import { createEmptySquareData } from "@/utils/pixelSquareDataCreator";
import axios from "axios";
import { Dotting, DottingRef, useBrush, useData, useHandlers } from "dotting";
import { get } from "http";
import Image from "next/image";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

const twentyEightByTwentyEight = createEmptySquareData(28);

export default function Home() {
  const getLabelResultOfWrittenCharacter = async (
    data: Array<Array<number>>
  ) => {
    const response = await axios.post("/api/predict/character", {
      data,
    });
    setResultProbabilities(response.data);
  };
  const [resultProbabilities, setResultProbabilities] =
    useState<Record<string, number>>();
  const ref = useRef<DottingRef>(null);
  const { changeBrushPattern } = useBrush(ref);
  const { dataArray } = useData(ref);

  useEffect(() => {
    changeBrushPattern([
      [0, 1, 0],
      [1, 1, 1],
      [0, 1, 0],
    ]);
  }, [ref.current]);

  const processedDataArray = useMemo(() => {
    return dataArray.map((data) => {
      return data.map((row) => {
        const rgb = hexToRgb(row.color);
        if (!rgb) {
          return 0;
        }

        return 255 - rgbToGrayscale(rgb?.r, rgb?.g, rgb?.b);
      });
    });
  }, [dataArray]);

  useEffect(() => {
    if (processedDataArray.length === 0) {
      return;
    }
    getLabelResultOfWrittenCharacter(processedDataArray);
  }, [processedDataArray]);

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="flex gap-10">
        <Dotting
          isGridFixed={true}
          ref={ref}
          width={500}
          height={500}
          initLayers={[
            {
              id: "square",
              data: twentyEightByTwentyEight,
            },
          ]}
          gridSquareLength={10}
          brushColor="#000000"
          defaultPixelColor="#ffffff"
        />
        <div className="flex flex-col">
          <div className="flex flex-col">
            Sent Data
            {processedDataArray.map((data, index) => {
              return (
                <div key={index} className="flex">
                  {data.map((row, rowIndex) => {
                    return (
                      <div
                        key={rowIndex}
                        className="h-1 w-1"
                        style={{
                          opacity: 1 - row,
                          backgroundColor: `rgb(${row}, ${row}, ${row})`,
                        }}
                      ></div>
                    );
                  })}
                </div>
              );
            })}
          </div>
          <div className="flex flex-col">
            <h2>Result</h2>
            {resultProbabilities && (
              <ProbabilityChart probabilities={resultProbabilities} />
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
