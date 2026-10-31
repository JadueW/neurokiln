import * as echarts from 'echarts/core'
import { BarChart, HeatmapChart, LineChart, ScatterChart } from 'echarts/charts'
import {
  GridComponent, LegendComponent, TooltipComponent, VisualMapComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

echarts.use([
  BarChart, HeatmapChart, LineChart, ScatterChart,
  GridComponent, LegendComponent, TooltipComponent, VisualMapComponent, CanvasRenderer,
])

export function createChart(el) {
  const chart = echarts.init(el)
  const observer = new ResizeObserver(() => chart.resize())
  observer.observe(el)
  return {
    chart,
    dispose() {
      observer.disconnect()
      chart.dispose()
    },
  }
}

export default echarts
