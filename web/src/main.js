import { createApp } from 'vue'
import {
  Aim,
  ArrowDown,
  CircleCheck,
  Close,
  Coin,
  Collection,
  Connection,
  Cpu,
  DataAnalysis,
  Delete,
  Document,
  Download,
  Expand,
  Files,
  Fold,
  Grid,
  Histogram,
  Link,
  MagicStick,
  Odometer,
  Plus,
  Rank,
  Refresh,
  Search,
  Setting,
  Shop,
  Timer,
  TrendCharts,
  Trophy,
  Upload,
  VideoPlay,
  View,
  Warning,
} from '@element-plus/icons-vue'
import App from './App.vue'
import i18n from './i18n'
import router from './router'
import './styles/index.css'

const icons = {
  Aim, ArrowDown, CircleCheck, Close, Coin, Collection, Connection, Cpu,
  DataAnalysis, Delete, Document, Download, Expand, Files, Fold, Grid,
  Histogram, Link, MagicStick, Odometer, Plus, Rank, Refresh, Search, Setting, Shop,
  Timer, TrendCharts, Trophy, Upload, VideoPlay, View, Warning,
}

const app = createApp(App)
for (const [name, component] of Object.entries(icons)) {
  app.component(name, component)
}
app.use(i18n)
app.use(router)
app.mount('#app')
