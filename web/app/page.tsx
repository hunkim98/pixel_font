"use client";

import { createEmptySquareData } from "@/utils/pixelSquareDataCreator";
import axios from "axios";
import { Dotting, DottingRef, useData, useHandlers } from "dotting";
import Image from "next/image";
import Link from "next/link";
import { useEffect, useRef } from "react";

const twentyEightByTwentyEight = createEmptySquareData(28);

export default function Home() {
  const getLabelResultOfWrittenCharacter = async () => {
    const response = await axios.post("/api/python", {});
  };
  const ref = useRef<DottingRef>(null);
  const { addStrokeEndListener, removeStrokeEndListener } = useHandlers(ref);
  const { dataArray } = useData(ref);

  useEffect(() => {
    const strokeEndListener = () => {
      getLabelResultOfWrittenCharacter();
    };
    addStrokeEndListener(strokeEndListener);
    return () => {
      removeStrokeEndListener(strokeEndListener);
    };
  }, [addStrokeEndListener, removeStrokeEndListener]);
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="flex gap-10">
        <Dotting
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
          brushColor="black"
        />
        <div>
          <button>Lable Result</button>
        </div>
      </div>
    </main>
  );
}
