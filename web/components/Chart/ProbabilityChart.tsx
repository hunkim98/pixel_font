import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

interface ProbabilityChartProps {
  // probabilities is an object with character keys and probability values
  probabilities: Record<string, number>;
}

const ProbabilityChart: React.FC<ProbabilityChartProps> = ({
  probabilities,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    // Create the D3 chart
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear the SVG container

    // Set the width and height of the SVG container
    const width = 400;
    const height = 200;
    const margin = { top: 10, right: 20, bottom: 40, left: 40 };

    // Calculate the inner width and height, considering margins
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    svg.attr("width", width).attr("height", height);

    // Convert probabilities object to an array of objects
    const data = Object.entries(probabilities)
      .map(([character, probability]) => ({
        character,
        probability,
      }))
      .sort((a, b) => b.probability - a.probability) // Sort by probability in descending order
      .slice(0, 5); // Select the top 5 probabilities

    // Create scales for x and y axes
    const xScale = d3
      .scaleBand()
      .domain(data.map((d) => d.character))
      .range([margin.left, innerWidth])
      .padding(0.1);
    const yScale = d3
      .scaleLinear()
      .domain([0, d3.max(data, (d) => d.probability)!])
      .nice()
      .range([innerHeight + margin.top, margin.top + margin.bottom]);

    // Create a color scale for bars
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    // Create a clipping path to keep the chart within the specified width and height
    svg
      .append("defs")
      .append("clipPath")
      .attr("id", "chart-clip")
      .append("rect")
      .attr("width", innerWidth)
      .attr("height", innerHeight);

    // Create a group for the chart content and apply the clipping path
    const chartGroup = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`)
      .attr("clip-path", "url(#chart-clip)");

    // Create and render bars for each character
    chartGroup
      .selectAll(".bar")
      .data(data)
      .enter()
      .append("rect")
      .attr("class", "bar")
      .attr("x", (d) => xScale(d.character)!)
      .attr("y", (d) => yScale(d.probability)!)
      .attr("width", xScale.bandwidth())
      .attr(
        "height",
        (d) => innerHeight - (yScale(d.probability)! - margin.top)
      )
      .attr("fill", (d, i) => colorScale(d.character));

    // Create and render text labels for values above the bars
    chartGroup
      .selectAll(".bar-label")
      .data(data)
      .enter()
      .append("text")
      .attr("class", "bar-label")
      .attr("x", (d) => xScale(d.character)! + xScale.bandwidth() / 2)
      .attr("y", (d) => yScale(d.probability)! - 5) // Adjust the vertical position
      .attr("text-anchor", "middle")
      .style("font-size", "16px") // Set the font size here
      .text((d) => d.probability.toFixed(2));

    // Create x-axis with character labels
    svg
      .append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(${margin.left},${height - margin.bottom})`)
      .style("font-size", "16px")
      .call(d3.axisBottom(xScale));

    // Create y-axis
    svg
      .append("g")
      .attr("class", "y-axis")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5));

    // Remove y-axis domain line
    svg.selectAll(".y-axis .domain").remove();
  }, [probabilities]);

  return (
    <div className="probability-chart">
      <h3>Top 5 Probabilities</h3>
      <svg ref={svgRef}></svg>
    </div>
  );
};

export default ProbabilityChart;
