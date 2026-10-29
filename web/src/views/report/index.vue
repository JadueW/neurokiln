<template>
  <div class="app-container">
    <h2 class="page-title">{{ $t('report.title') }}</h2>
    <p class="page-desc">{{ $t('report.description') }}</p>

    <div class="stat-grid" style="margin-bottom:16px">
      <el-card v-for="card in statCards" :key="card.label" shadow="never">
        <div class="stat-card">
          <div class="stat-icon" :style="{ background: card.color }">
            <el-icon><component :is="card.icon" /></el-icon>
          </div>
          <div>
            <div class="stat-value">{{ card.value }}</div>
            <div class="stat-label">{{ card.label }}</div>
          </div>
        </div>
      </el-card>
    </div>

    <el-card shadow="never">
      <div class="filter-bar">
        <el-input v-model="keyword" :placeholder="$t('report.searchPlaceholder')" clearable style="width:240px" prefix-icon="Search" />
        <el-select v-model="status" :placeholder="$t('report.statusPlaceholder')" clearable style="width:130px">
          <el-option :label="$t('report.completed')" value="completed" />
          <el-option :label="$t('report.failed')" value="failed" />
        </el-select>
        <el-button icon="Refresh" @click="refresh">{{ $t('common.refresh') }}</el-button>
        <span class="text-muted" style="font-size:12px">{{ $t('report.database', { path: stats.storage || '-' }) }}</span>
      </div>
      <el-table :data="filtered" border stripe v-loading="loading" @row-dblclick="openDetail">
        <el-table-column prop="id" label="#" width="60" align="center" />
        <el-table-column :label="$t('report.experimentColumn')" min-width="240">
          <template #default="{ row }">
            <el-link type="primary" @click="openDetail(row)">{{ tr(row.name) }}</el-link>
            <div class="text-muted" style="font-size:12px">{{ row.protocol }} · {{ row.n_runs }} runs</div>
          </template>
        </el-table-column>
        <el-table-column :label="$t('report.datasetColumn')" min-width="170">
          <template #default="{ row }">
            {{ tr(row.dataset.name) }}
            <div class="mono text-muted" style="font-size:11px">{{ row.dataset.signature.slice(0, 12) }}</div>
          </template>
        </el-table-column>
        <el-table-column :label="$t('report.bestPipeline')" min-width="170">
          <template #default="{ row }">{{ pipelineName(row.best?.pipeline) }}</template>
        </el-table-column>
        <el-table-column :label="$t('report.bestBA')" width="100" align="center">
          <template #default="{ row }">
            <b v-if="row.best?.balanced_accuracy !== null">{{ formatMetric(row.best?.balanced_accuracy) }}</b>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column :label="$t('report.statusColumn')" width="100" align="center">
          <template #default="{ row }">
            <el-tag size="small" :type="statusType(row.status)">{{ statusLabel(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="$t('report.createdAt')" width="170" align="center">
          <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
        </el-table-column>
        <el-table-column :label="$t('report.actions')" width="150" align="center" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" @click="openDetail(row)">{{ $t('report.detail') }}</el-button>
            <el-button link type="danger" @click="removeExperiment(row)">{{ $t('common.delete') }}</el-button>
          </template>
        </el-table-column>
        <template #empty>
          <el-empty :description="$t('report.emptyExperiments')" />
        </template>
      </el-table>
    </el-card>

    <el-drawer
      v-model="drawerVisible"
      :title="detail ? $t('report.drawerTitleWith', { id: detail.id, name: tr(detail.name) }) : $t('report.drawerTitle')"
      size="min(1040px, 92%)"
      @opened="renderCharts"
      @closed="disposeCharts"
    >
      <template v-if="detail">
        <el-alert
          type="success"
          :closable="false"
          style="margin-bottom:14px"
          :title="$t('report.summaryAlert', { pipeline: pipelineName(detail.summary?.best_pipeline), ba: formatMetric(detail.summary?.best_balanced_accuracy), runs: detail.n_runs, duration: formatDuration(detail.duration_ms) })"
        />
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item :label="$t('report.datasetColumn')">{{ $t('report.datasetValue', { name: tr(detail.dataset.name), adapter: detail.dataset.manifest?.adapter || '-' }) }}</el-descriptions-item>
          <el-descriptions-item :label="$t('report.protocol')">{{ detail.protocol }} · folds={{ detail.protocol_params?.folds }} · seeds={{ detail.seeds.join(',') }}</el-descriptions-item>
          <el-descriptions-item :label="$t('report.dataSignature')" :span="2"><span class="mono signature">{{ detail.dataset.signature }}</span></el-descriptions-item>
          <el-descriptions-item :label="$t('report.engineVersion')">{{ detail.engine_version }}</el-descriptions-item>
          <el-descriptions-item :label="$t('report.trialsClasses')">{{ detail.dataset.manifest?.loaded_n_trials }} / {{ detail.dataset.manifest?.loaded_classes }}</el-descriptions-item>
          <el-descriptions-item :label="$t('report.subjects')">{{ (detail.dataset.manifest?.subjects || []).join(', ') }}</el-descriptions-item>
          <el-descriptions-item :label="$t('report.createdAt')">{{ formatTime(detail.created_at) }}</el-descriptions-item>
          <el-descriptions-item :label="$t('report.completedAt')">{{ formatTime(detail.completed_at) }}</el-descriptions-item>
          <el-descriptions-item :label="$t('report.notes')" :span="2">{{ tr(detail.notes) || '-' }}</el-descriptions-item>
        </el-descriptions>

        <el-row :gutter="16" style="margin-top:16px">
          <el-col :xs="24" :md="12" style="margin-bottom:16px">
            <el-card shadow="never">
              <template #header><b>{{ $t('report.avgMetricsTitle') }}</b></template>
              <div ref="barEl" style="height:300px;width:100%" />
            </el-card>
          </el-col>
          <el-col :xs="24" :md="12" style="margin-bottom:16px">
            <el-card shadow="never">
              <template #header><b>{{ $t('report.foldBATitle') }}</b></template>
              <div ref="lineEl" style="height:300px;width:100%" />
            </el-card>
          </el-col>
        </el-row>

        <el-card v-if="calibrationRows.length" shadow="never" style="margin-bottom:16px">
          <template #header>
            <div class="section-head">
              <b>{{ $t('report.calibrationTitle') }}</b>
              <el-select v-model="calibrationPipeline" size="small" style="width:230px" @change="renderReliability">
                <el-option v-for="row in calibrationRows" :key="row.pipeline" :label="pipelineName(row.pipeline)" :value="row.pipeline" />
              </el-select>
            </div>
          </template>
          <el-row :gutter="16">
            <el-col :xs="24" :md="12" style="margin-bottom:12px">
              <div ref="reliabilityEl" style="height:300px;width:100%" />
            </el-col>
            <el-col :xs="24" :md="12" style="margin-bottom:12px">
              <el-table :data="calibrationRows" border size="small">
                <el-table-column :label="$t('report.pipelineColumn')" min-width="150">
                  <template #default="{ row }">{{ pipelineName(row.pipeline) }}</template>
                </el-table-column>
                <el-table-column label="ECE" width="80" align="center">
                  <template #default="{ row }">{{ formatMetric(row.ece) }}</template>
                </el-table-column>
                <el-table-column label="MCE" width="80" align="center">
                  <template #default="{ row }">{{ formatMetric(row.mce) }}</template>
                </el-table-column>
                <el-table-column label="Brier" width="80" align="center">
                  <template #default="{ row }">{{ formatMetric(row.brier) }}</template>
                </el-table-column>
                <el-table-column label="NLL" width="80" align="center">
                  <template #default="{ row }">{{ formatMetric(row.nll) }}</template>
                </el-table-column>
                <el-table-column :label="$t('report.confidenceAccuracy')" width="120" align="center">
                  <template #default="{ row }">{{ formatMetric(row.confidence_mean, 3) }} / {{ formatMetric(row.accuracy, 3) }}</template>
                </el-table-column>
                <el-table-column prop="n_evaluated" :label="$t('report.trials')" width="70" align="center" />
              </el-table>
              <div class="stat-note">
                {{ $t('report.calibrationNotePrefix') }}<span class="mono">predict_proba</span>{{ $t('report.calibrationNoteSuffix') }}
              </div>
            </el-col>
          </el-row>
        </el-card>

        <el-card shadow="never" style="margin-bottom:16px">
          <template #header><b>{{ $t('report.foldResultsTitle') }}</b></template>
          <el-table :data="sortedRows" border stripe>
            <el-table-column :label="$t('report.pipelineColumn')" min-width="180">
              <template #default="{ row }">{{ pipelineName(row.pipeline) }}</template>
            </el-table-column>
            <el-table-column :label="$t('report.baMeanSd')" min-width="170" align="center">
              <template #default="{ row }">
                <b>{{ formatMetric(row.balanced_accuracy_mean) }}</b>
                <span class="text-muted"> ± {{ formatMetric(row.balanced_accuracy_std) }}</span>
              </template>
            </el-table-column>
            <el-table-column label="95% CI" min-width="140" align="center">
              <template #default="{ row }">{{ formatCi(row.balanced_accuracy_ci95 || ciByPipeline[row.pipeline]) }}</template>
            </el-table-column>
            <el-table-column label="Kappa" width="100" align="center">
              <template #default="{ row }">{{ formatMetric(row.cohen_kappa_mean) }}</template>
            </el-table-column>
            <el-table-column label="ECE" width="90" align="center">
              <template #default="{ row }">
                <el-tooltip v-if="row.calibration?.available" :content="`Brier ${formatMetric(row.calibration.brier)} · NLL ${formatMetric(row.calibration.nll)}`" placement="top">
                  <span>{{ formatMetric(row.calibration.ece) }}</span>
                </el-tooltip>
                <span v-else class="text-muted">-</span>
              </template>
            </el-table-column>
            <el-table-column :label="$t('report.avgDuration')" width="110" align="center">
              <template #default="{ row }">{{ formatDuration(row.duration_ms_mean) }}</template>
            </el-table-column>
            <el-table-column prop="n_splits" :label="$t('report.splitCount')" width="90" align="center" />
            <el-table-column type="expand">
              <template #default="{ row }">
                <el-table :data="row.splits" size="small" border style="margin:8px 16px">
                  <el-table-column prop="split.fold" :label="$t('report.fold')" width="60" align="center" />
                  <el-table-column prop="split.repeat" :label="$t('report.repeat')" width="70" align="center" />
                  <el-table-column prop="split.seed" :label="$t('report.seed')" width="70" align="center" />
                  <el-table-column label="BA" width="100" align="center">
                    <template #default="{ row: split }">{{ formatMetric(split.balanced_accuracy) }}</template>
                  </el-table-column>
                  <el-table-column label="Kappa" width="100" align="center">
                    <template #default="{ row: split }">{{ formatMetric(split.cohen_kappa) }}</template>
                  </el-table-column>
                  <el-table-column label="ECE" width="90" align="center">
                    <template #default="{ row: split }">
                      {{ split.calibration?.available ? formatMetric(split.calibration.ece) : '-' }}
                    </template>
                  </el-table-column>
                  <el-table-column :label="$t('report.duration')" width="100" align="center">
                    <template #default="{ row: split }">{{ formatDuration(split.duration_ms) }}</template>
                  </el-table-column>
                  <el-table-column prop="split.signature" :label="$t('report.splitSignature')" align="center" />
                </el-table>
              </template>
            </el-table-column>
          </el-table>
        </el-card>

        <el-card v-if="confusion" shadow="never" style="margin-bottom:16px">
          <template #header><b>{{ $t('report.confusionTitle', { pipeline: pipelineName(detail.summary?.best_pipeline) }) }}</b></template>
          <el-table :data="confusion.rows" border size="small">
            <el-table-column prop="label" width="130">
              <template #header>{{ $t('report.trueLabel') }} \\ {{ $t('report.predictedLabel') }}</template>
            </el-table-column>
            <el-table-column v-for="(name, index) in confusion.labels" :key="name" :label="name" align="center">
              <template #default="{ row }">{{ row.cells[index] }}</template>
            </el-table-column>
          </el-table>
        </el-card>

        <el-card v-if="statistics" shadow="never" style="margin-bottom:16px">
          <template #header>
            <div class="section-head">
              <b>{{ $t('report.statisticsTitle') }}</b>
              <span class="text-muted">
                {{ statistics.unit === 'subject' ? $t('report.subjectLevel') : $t('report.foldLevel') }} Balanced Accuracy ·
                n = {{ statistics.n_units }} · α = {{ statistics.alpha }}
              </span>
            </div>
          </template>
          <el-alert type="info" :closable="false" :title="tr(statistics.note)" />
          <el-row :gutter="16" style="margin-top:12px">
            <el-col :xs="24" :md="12" style="margin-bottom:12px">
              <div class="section-title">{{ $t('report.rankFriedmanTitle') }}</div>
              <el-table :data="rankRows" border size="small">
                <el-table-column :label="$t('report.rank')" width="60" align="center">
                  <template #default="{ $index }">{{ $index + 1 }}</template>
                </el-table-column>
                <el-table-column :label="$t('report.pipelineColumn')" min-width="140">
                  <template #default="{ row }">{{ pipelineName(row.pipeline) }}</template>
                </el-table-column>
                <el-table-column :label="$t('report.meanRank')" width="90" align="center">
                  <template #default="{ row }">{{ formatMetric(row.mean_rank, 2) }}</template>
                </el-table-column>
                <el-table-column :label="$t('report.meanBA')" width="90" align="center">
                  <template #default="{ row }">{{ formatMetric(row.mean_balanced_accuracy) }}</template>
                </el-table-column>
                <template #empty><span class="text-muted">{{ $t('report.noTestUnits') }}</span></template>
              </el-table>
              <div class="stat-note">
                <template v-if="statistics.friedman?.available">
                  {{ $t('report.friedmanLine', {
                    statistic: formatMetric(statistics.friedman.statistic, 3),
                    p: formatP(statistics.friedman.p_value),
                    conclusion: statistics.friedman.significant ? $t('report.significantDifference') : $t('report.noSignificantDifference'),
                  }) }}
                </template>
                <template v-else>{{ $t('report.friedmanFallback', { reason: tr(statistics.friedman?.reason) || $t('report.friedmanUnavailable') }) }}</template>
                <span v-if="statistics.nemenyi?.available" class="nemenyi-note">
                  {{ $t('report.nemenyiLine', { value: formatMetric(statistics.nemenyi.critical_difference, 3) }) }}
                </span>
              </div>
            </el-col>
            <el-col :xs="24" :md="12" style="margin-bottom:12px">
              <div class="section-title">{{ $t('report.pairwiseTitle') }}</div>
              <el-table :data="pairwiseRows" border size="small" max-height="260">
                <el-table-column :label="$t('report.comparison')" min-width="170">
                  <template #default="{ row }">
                    {{ pipelineName(row.pipeline_a) }} <span class="text-muted">vs</span>
                    {{ pipelineName(row.pipeline_b) }}
                  </template>
                </el-table-column>
                <el-table-column label="ΔBA" width="86" align="center">
                  <template #default="{ row }">{{ formatSigned(row.mean_diff) }}</template>
                </el-table-column>
                <el-table-column label="p (Holm)" width="96" align="center">
                  <template #default="{ row }">{{ formatP(row.p_value_adjusted) }}</template>
                </el-table-column>
                <el-table-column :label="$t('report.significant')" width="70" align="center">
                  <template #default="{ row }">
                    <el-tag size="small" :type="row.significant ? 'danger' : 'info'">{{ row.significant ? $t('report.yes') : $t('report.no') }}</el-tag>
                  </template>
                </el-table-column>
                <template #empty><span class="text-muted">{{ $t('report.minPipelines') }}</span></template>
              </el-table>
            </el-col>
          </el-row>
        </el-card>

        <div style="margin-top:16px;display:flex;flex-wrap:wrap;gap:10px">
          <el-button type="primary" icon="View" @click="previewReport">{{ $t('report.previewHtml') }}</el-button>
          <el-button type="primary" plain icon="Download" @click="downloadReport('html')">{{ $t('report.exportHtml') }}</el-button>
          <el-button type="success" plain icon="Download" @click="downloadReport('xlsx')">{{ $t('report.exportExcel') }}</el-button>
          <el-button type="warning" plain icon="Document" @click="downloadReport('docx')">{{ $t('report.exportWord') }}</el-button>
          <el-button type="info" plain icon="Document" @click="downloadReport('tex')">{{ $t('report.exportLatex') }}</el-button>
          <el-button icon="Share" @click="openShareCard">{{ $t('report.shareCard') }}</el-button>
          <el-button icon="Download" @click="exportDetail">{{ $t('report.exportJson') }}</el-button>
          <el-button type="danger" plain icon="Delete" @click="removeExperiment(detail)">{{ $t('report.deleteExperiment') }}</el-button>
        </div>
      </template>
    </el-drawer>

    <el-dialog
      v-model="previewVisible"
      :title="detail ? $t('report.previewTitleWith', { id: detail.id }) : $t('report.previewTitle')"
      width="min(1100px, 94%)"
      top="4vh"
      destroy-on-close
      @closed="releasePreview"
    >
      <div v-loading="previewLoading" class="preview-body">
        <iframe v-if="previewUrl" :src="previewUrl" :title="$t('report.previewTitle')" sandbox="allow-same-origin" />
      </div>
      <template #footer>
        <el-button @click="previewVisible = false">{{ $t('report.close') }}</el-button>
        <el-button type="primary" @click="downloadReport('html')">{{ $t('report.downloadHtml') }}</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="shareVisible" :title="$t('report.shareCard')" width="min(900px, 94%)" destroy-on-close>
      <div v-loading="shareLoading" class="share-preview">
        <img v-if="shareImage" :src="shareImage" :alt="$t('report.shareCardAlt')" />
        <p v-else class="text-muted">{{ $t('report.shareCardLoading') }}</p>
      </div>
      <template #footer>
        <el-button @click="shareVisible = false">{{ $t('report.close') }}</el-button>
        <el-button type="primary" icon="Download" :disabled="!shareImage" @click="saveShareCard">{{ $t('report.downloadPng') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { tr } from '@/i18n/catalog'
import { useRoute } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { deleteExperiment, getExperiment, getExperimentStats, getExperiments, getPipelines } from '@/api'
import { createChart } from '@/utils/chart'
import { downloadWithAuth, fetchWithAuth } from '@/utils/download'
import { downloadJson, formatDuration, formatMetric, formatTime } from '@/utils/format'
import { buildShareCard, downloadShareCard } from '@/utils/share-card'

const { t } = useI18n()

const SHORT_LABELS = {
  raw_centroid: 'Raw',
  centered_centroid: 'Center',
  var_centroid: 'Var',
  csp_lda: 'CSP+LDA',
  fbcsp_lda: 'FBCSP',
  riemann_mdm: 'MDM',
  ts_lr: 'TS+LR',
  eegnet: 'EEGNet',
  shallowconvnet: 'Shallow',
  xdawn_lda: 'xDAWN',
  ssvep_cca: 'CCA',
  ssvep_fbcca: 'FBCCA',
}

const route = useRoute()
const items = ref([])
const pipelines = ref([])
const stats = ref({})
const keyword = ref('')
const status = ref('')
const loading = ref(false)
const detail = ref(null)
const drawerVisible = ref(false)
const previewVisible = ref(false)
const previewLoading = ref(false)
const previewUrl = ref('')
const shareVisible = ref(false)
const shareLoading = ref(false)
const shareImage = ref('')
const barEl = ref(null)
const lineEl = ref(null)
const reliabilityEl = ref(null)
const calibrationPipeline = ref('')
let barCtl = null
let lineCtl = null
let reliabilityCtl = null

const statCards = computed(() => [
  { label: t('report.totalExperiments'), value: stats.value.experiments ?? '-', icon: 'DataAnalysis', color: '#409eff' },
  { label: t('report.completed'), value: stats.value.completed ?? '-', icon: 'CircleCheck', color: '#67c23a' },
  { label: t('report.evalRuns'), value: stats.value.runs ?? '-', icon: 'Histogram', color: '#e6a23c' },
  { label: t('report.datasetsUsed'), value: stats.value.datasets_used ?? '-', icon: 'Coin', color: '#9254de' },
  {
    label: t('report.bestBA'),
    value: stats.value.best_run ? formatMetric(stats.value.best_run.balanced_accuracy) : '-',
    icon: 'Trophy',
    color: '#f56c6c',
  },
])

const filtered = computed(() => items.value.filter((item) => {
  const text = `${item.name}${tr(item.name)}${item.dataset?.name}${tr(item.dataset?.name)}${item.protocol}`.toLowerCase()
  return (!keyword.value || text.includes(keyword.value.toLowerCase()))
    && (!status.value || item.status === status.value)
}))

const sortedRows = computed(() => {
  if (!detail.value) return []
  return [...detail.value.rows].sort(
    (left, right) => (right.balanced_accuracy_mean ?? 0) - (left.balanced_accuracy_mean ?? 0),
  )
})

const confusion = computed(() => {
  if (!detail.value || !sortedRows.value.length) return null
  const best = sortedRows.value[0]
  const labelNames = detail.value.dataset.manifest?.label_names || []
  let size = Math.max(labelNames.length, 1)
  for (const split of best.splits) {
    for (const value of [...split.y_true, ...split.y_pred]) size = Math.max(size, value + 1)
  }
  const matrix = Array.from({ length: size }, () => Array(size).fill(0))
  for (const split of best.splits) {
    split.y_true.forEach((truth, index) => {
      const predicted = split.y_pred[index]
      if (matrix[truth] && matrix[truth][predicted] !== undefined) matrix[truth][predicted] += 1
    })
  }
  const labels = Array.from({ length: size }, (_, index) => labelNames[index] || `class_${index}`)
  return {
    labels,
    rows: labels.map((label, index) => ({ label, cells: matrix[index] })),
  }
})

const statistics = computed(() => detail.value?.statistics || null)
const rankRows = computed(() => statistics.value?.average_ranks?.ranks || [])
const pairwiseRows = computed(() => statistics.value?.pairwise || [])
const ciByPipeline = computed(() => {
  const pipelines = detail.value?.summary?.pipelines || []
  return Object.fromEntries(
    pipelines.map((item) => [item.pipeline, item.balanced_accuracy_ci95]),
  )
})
const calibrationRows = computed(() => sortedRows.value
  .filter((row) => row.calibration?.available)
  .map((row) => ({ pipeline: row.pipeline, ...row.calibration })))

function formatCi(value) {
  if (!Array.isArray(value) || value.length !== 2) return '-'
  if (value.some((item) => item === null || item === undefined)) return '-'
  return `[${Number(value[0]).toFixed(3)}, ${Number(value[1]).toFixed(3)}]`
}

function formatSigned(value) {
  if (value === null || value === undefined) return '-'
  const number = Number(value)
  return `${number >= 0 ? '+' : ''}${number.toFixed(4)}`
}

function formatP(value) {
  if (value === null || value === undefined) return '-'
  const number = Number(value)
  if (!Number.isFinite(number)) return '-'
  return number < 0.0001 ? '< 0.0001' : number.toFixed(4)
}

function pipelineName(id) {
  if (!id) return '-'
  return tr(detail.value?.rows.find((row) => row.pipeline === id)?.pipeline_name)
    || tr(pipelines.value.find((item) => item.id === id)?.name)
    || id
}

function statusLabel(value) {
  return { completed: t('report.completed'), failed: t('report.failed'), running: t('report.running') }[value] || value
}

function statusType(value) {
  return { completed: 'success', failed: 'danger', running: 'warning' }[value] || 'info'
}

async function refresh() {
  loading.value = true
  try {
    const [list, statsResponse, pipelineResponse] = await Promise.all([
      getExperiments({ limit: 100 }),
      getExperimentStats(),
      pipelines.value.length ? Promise.resolve({ items: pipelines.value }) : getPipelines(),
    ])
    items.value = list.items
    stats.value = statsResponse
    pipelines.value = pipelineResponse.items
  } finally {
    loading.value = false
  }
}

async function openDetail(row) {
  detail.value = await getExperiment(row.id)
  calibrationPipeline.value = calibrationRows.value[0]?.pipeline || ''
  drawerVisible.value = true
  await nextTick()
  if (drawerVisible.value) renderCharts()
}

function disposeCharts() {
  barCtl?.dispose()
  lineCtl?.dispose()
  reliabilityCtl?.dispose()
  barCtl = null
  lineCtl = null
  reliabilityCtl = null
}

function renderCharts() {
  if (!detail.value) return
  const rows = sortedRows.value
  if (barEl.value) {
    if (!barCtl) barCtl = createChart(barEl.value)
    barCtl.chart.setOption({
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { top: 0, data: ['Balanced Accuracy', 'Cohen Kappa'] },
      grid: { left: 12, right: 16, top: 44, bottom: 16, containLabel: true },
      xAxis: {
        type: 'category',
        data: rows.map((row) => SHORT_LABELS[row.pipeline] || row.pipeline),
        axisLabel: { interval: 0, fontSize: 11, color: '#606266' },
        axisTick: { show: false },
      },
      yAxis: { type: 'value', min: 0, max: 1, axisLabel: { color: '#909399' }, splitLine: { lineStyle: { type: 'dashed' } } },
      series: [
        {
          name: 'Balanced Accuracy', type: 'bar', barMaxWidth: 26,
          data: rows.map((row) => +row.balanced_accuracy_mean.toFixed(4)),
          itemStyle: { color: '#409eff', borderRadius: [3, 3, 0, 0] },
        },
        {
          name: 'Cohen Kappa', type: 'bar', barMaxWidth: 26,
          data: rows.map((row) => +row.cohen_kappa_mean.toFixed(4)),
          itemStyle: { color: '#67c23a', borderRadius: [3, 3, 0, 0] },
        },
      ],
    }, true)
  }
  if (lineEl.value) {
    if (!lineCtl) lineCtl = createChart(lineEl.value)
    const length = Math.max(0, ...rows.map((row) => row.splits.length))
    lineCtl.chart.setOption({
      tooltip: { trigger: 'axis' },
      legend: { top: 0, type: 'scroll' },
      grid: { left: 12, right: 16, top: 44, bottom: 16, containLabel: true },
      xAxis: {
        type: 'category',
        data: Array.from({ length }, (_, index) => `#${index + 1}`),
        axisLabel: { color: '#606266' },
      },
      yAxis: { type: 'value', min: 0, max: 1, axisLabel: { color: '#909399' }, splitLine: { lineStyle: { type: 'dashed' } } },
      series: rows.map((row) => ({
        name: SHORT_LABELS[row.pipeline] || row.pipeline,
        type: 'line',
        smooth: true,
        symbolSize: 6,
        data: row.splits.map((split) => +split.balanced_accuracy.toFixed(4)),
      })),
    }, true)
  }
  renderReliability()
}

function renderReliability() {
  if (!reliabilityEl.value || !calibrationRows.value.length) return
  if (!calibrationPipeline.value || !calibrationRows.value.some((row) => row.pipeline === calibrationPipeline.value)) {
    calibrationPipeline.value = calibrationRows.value[0].pipeline
  }
  const row = calibrationRows.value.find((item) => item.pipeline === calibrationPipeline.value)
  const curve = [...(row?.reliability || [])].sort((left, right) => left.confidence - right.confidence)
  const points = curve.map((item) => [item.confidence, item.accuracy, item.count])
  if (!reliabilityCtl) reliabilityCtl = createChart(reliabilityEl.value)
  reliabilityCtl.chart.setOption({
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        if (params.seriesName === t('report.idealDiagonal')) return t('report.idealCalibrationTooltip')
        const [confidence, accuracy, count] = params.value || []
        return `${pipelineName(row?.pipeline)}<br/>${t('report.confidence')} ${Number(confidence).toFixed(3)}<br/>${t('report.accuracy')} ${Number(accuracy).toFixed(3)}<br/>${t('report.trials')} ${count ?? '-'}`
      },
    },
    legend: { top: 0, data: [t('report.actualAccuracy'), t('report.binnedTrials'), t('report.idealDiagonal')] },
    grid: { left: 12, right: 20, top: 44, bottom: 12, containLabel: true },
    xAxis: {
      type: 'value', min: 0, max: 1, name: t('report.predictedConfidence'),
      axisLabel: { color: '#606266' }, splitLine: { lineStyle: { type: 'dashed' } },
    },
    yAxis: {
      type: 'value', min: 0, max: 1, name: t('report.actualAccuracy'),
      axisLabel: { color: '#606266' }, splitLine: { lineStyle: { type: 'dashed' } },
    },
    series: [
      {
        name: t('report.idealDiagonal'), type: 'line', data: [[0, 0], [1, 1]], showSymbol: false,
        lineStyle: { type: 'dashed', color: '#c0c4cc', width: 1.5 }, silent: true,
      },
      {
        name: t('report.binnedTrials'), type: 'scatter', data: points,
        symbolSize: (value) => Math.max(7, Math.min(26, Math.sqrt(value[2] || 1) * 1.8)),
        itemStyle: { color: 'rgba(64,158,255,0.18)' },
      },
      {
        name: t('report.actualAccuracy'), type: 'line', data: points, symbolSize: 6,
        lineStyle: { color: '#409eff', width: 2 }, itemStyle: { color: '#409eff' },
      },
    ],
  }, true)
}

function exportDetail() {
  downloadJson(`experiment-${detail.value.id}.json`, detail.value)
}

async function openShareCard() {
  if (!detail.value) return
  shareVisible.value = true
  shareLoading.value = true
  shareImage.value = ''
  try {
    shareImage.value = await buildShareCard(detail.value)
  } catch {
    ElMessage.error(t('report.shareCardFailed'))
  } finally {
    shareLoading.value = false
  }
}

function saveShareCard() {
  if (!shareImage.value || !detail.value) return
  downloadShareCard(shareImage.value, detail.value.id)
}

async function downloadReport(format) {
  if (!detail.value) return
  try {
    await downloadWithAuth(
      `/api/experiments/${detail.value.id}/report.${format}`,
      `neurokiln-experiment-${detail.value.id}.${format}`,
    )
    ElMessage.success(t('report.downloadStarted'))
  } catch (error) {
    ElMessage.error(error.message)
  }
}

function releasePreview() {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = ''
  }
}

async function previewReport() {
  if (!detail.value) return
  previewLoading.value = true
  previewVisible.value = true
  try {
    const blob = await fetchWithAuth(`/api/experiments/${detail.value.id}/report.html`)
    releasePreview()
    previewUrl.value = URL.createObjectURL(blob)
  } catch (error) {
    previewVisible.value = false
    ElMessage.error(error.message)
  } finally {
    previewLoading.value = false
  }
}

async function removeExperiment(row) {
  try {
    await ElMessageBox.confirm(t('report.deleteConfirm', { name: tr(row.name) }), t('report.deleteExperiment'), {
      type: 'warning', confirmButtonText: t('common.delete'), cancelButtonText: t('common.cancel'),
    })
  } catch {
    return
  }
  await deleteExperiment(row.id)
  ElMessage.success(t('report.deleted'))
  drawerVisible.value = false
  detail.value = null
  await refresh()
}

onMounted(async () => {
  await refresh()
  const experimentId = Number(route.query.experiment)
  if (Number.isFinite(experimentId) && experimentId > 0) {
    await openDetail({ id: experimentId })
  }
})

onBeforeUnmount(() => {
  disposeCharts()
  releasePreview()
})
</script>

<style scoped>
.signature { word-break: break-all; }
.preview-body { height: min(78vh, 900px); }
.preview-body iframe { width: 100%; height: 100%; border: 0; background: #fff; }
.share-preview { display: flex; justify-content: center; align-items: center; min-height: 260px; }
.share-preview img { max-width: 100%; border-radius: 12px; box-shadow: 0 14px 36px rgba(15, 35, 60, 0.22); }
.section-head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; }
.section-title { font-size: 13px; font-weight: 600; margin-bottom: 8px; }
.stat-note { font-size: 12px; line-height: 1.8; color: #606266; margin-top: 8px; }
.nemenyi-note { margin-left: 6px; }
</style>
