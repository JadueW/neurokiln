import * as echarts from 'echarts/core'
import { BarChart, LineChart, ScatterChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

echarts.use([
  BarChart, LineChart, ScatterChart,
  GridComponent, LegendComponent, TooltipComponent, CanvasRenderer,
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
