<template>
  <div class="app-container">
    <h2 class="page-title">{{ $t('arena.title') }}</h2>
    <p class="page-desc">{{ $t('arena.description') }}</p>

    <el-card shadow="never" style="margin-bottom:16px">
      <el-form :inline="true" label-width="92px">
        <el-form-item :label="$t('arena.dataset')">
          <el-select :model-value="selection.datasetId" style="width:240px" @change="onDatasetChange">
            <el-option v-for="item in datasets" :key="item.id" :label="tr(item.name)" :value="item.id" :disabled="!item.ready">
              <span>{{ tr(item.name) }}</span>
              <el-tag size="small" type="info" style="float:right;margin-left:12px">{{ item.paradigm }}</el-tag>
            </el-option>
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('arena.subjects')">
          <el-select v-model="selection.subjectIds" multiple collapse-tags clearable style="width:170px" :placeholder="$t('arena.defaultSubjects')">
            <el-option v-for="subject in subjectOptions" :key="subject" :label="subject" :value="subject" />
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('arena.protocol')">
          <el-select v-model="form.protocol" style="width:180px">
            <el-option :label="$t('arena.protocolWithinSubject')" value="within_subject" />
            <el-option :label="$t('arena.protocolLoso')" value="loso" />
            <el-option :label="$t('arena.protocolCrossSession')" value="cross_session" />
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('arena.folds')" v-if="form.protocol === 'within_subject'"><el-input-number v-model="form.folds" :min="2" :max="10" /></el-form-item>
        <el-form-item :label="$t('arena.repeats')" v-if="form.protocol === 'within_subject'"><el-input-number v-model="form.repeats" :min="1" :max="5" /></el-form-item>
        <el-form-item :label="$t('arena.seed')" v-if="form.protocol === 'within_subject'"><el-input v-model="seedText" style="width:120px" /></el-form-item>
        <el-form-item :label="$t('arena.classCount')" v-if="classOptions.length">
          <el-select v-model="form.classes" style="width:100px">
            <el-option v-for="count in classOptions" :key="count" :label="count" :value="count" />
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('arena.trialsPerClass')" v-if="isSynthetic"><el-input-number v-model="form.trialsPerClass" :min="10" :max="200" :step="10" /></el-form-item>
        <el-form-item :label="$t('arena.sessions')" v-if="isSynthetic && form.protocol === 'cross_session'"><el-input-number v-model="form.sessions" :min="2" :max="4" /></el-form-item>
        <el-form-item :label="$t('arena.pipeline')">
          <el-select v-model="form.pipelines" multiple collapse-tags style="width:380px" :placeholder="$t('arena.pipelinePlaceholder')">
            <el-option v-for="item in compatiblePipelines" :key="item.id" :label="tr(item.name)" :value="item.id">
              <span>{{ tr(item.name) }}</span>
              <el-tag size="small" type="info" style="float:right;margin-left:12px">{{ item.paradigm || 'MI' }}</el-tag>
            </el-option>
            <template #footer>
              <div class="text-muted" style="font-size:12px;padding:0 8px">
                {{ $t('arena.compatibleHint', { paradigm: current?.paradigm || 'MI', compatible: compatiblePipelines.length, total: pipelines.length }) }}
              </div>
            </template>
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('arena.experimentName')">
          <el-input v-model="form.name" :placeholder="$t('arena.experimentNamePlaceholder')" style="width:200px" />
        </el-form-item>
        <el-form-item :label="$t('arena.saveToDb')">
          <el-switch v-model="form.save" />
        </el-form-item>
        <el-form-item :label="$t('arena.backgroundQueue')">
          <el-tooltip :content="$t('arena.queueHint')" placement="top">
            <el-switch v-model="form.async" />
          </el-tooltip>
        </el-form-item>
        <el-form-item :label="$t('arena.preprocessCache')">
          <el-tooltip :content="$t('arena.cacheHint')" placement="top">
            <el-switch v-model="form.useCache" />
          </el-tooltip>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" icon="VideoPlay" :loading="submitting" :disabled="jobActive" @click="run">{{ $t('arena.runArena') }}</el-button>
          <el-button icon="Timer" @click="openJobs">{{ $t('arena.jobQueue') }}</el-button>
        </el-form-item>
      </el-form>
      <el-alert v-if="current?.requires_download" type="info" :closable="false" style="margin-bottom:8px"
        :title="$t('arena.downloadHint', { name: current.name })" />
      <el-alert v-if="hasDeep" type="warning" :closable="false"
        :title="$t('arena.deepWarning')" />
    </el-card>

    <el-card v-if="job" shadow="never" style="margin-bottom:16px">
      <div class="job-row">
        <div class="job-title">
          <b>{{ $t('arena.backgroundTask') }}</b>
          <span class="mono text-muted">{{ job.id.slice(0, 8) }}</span>
          <el-tag size="small" :type="jobStatusType">{{ jobStatusLabel }}</el-tag>
          <span v-if="job.request?.pipelines?.length" class="text-muted job-pipelines">
            {{ job.request.pipelines.map((id) => pipelineName(id)).join(' vs ') }}
          </span>
        </div>
        <div class="job-actions">
          <el-button v-if="jobActive" size="small" type="danger" plain icon="Close" @click="cancelActiveJob">{{ $t('arena.cancelJob') }}</el-button>
          <el-button v-if="job.status === 'completed' && result" size="small" type="primary" @click="applyResult(job.result)">{{ $t('arena.viewResult') }}</el-button>
          <el-button v-if="!jobActive" size="small" @click="job = null">{{ $t('arena.collapse') }}</el-button>
        </div>
      </div>
      <el-progress
        :percentage="jobProgress"
        :status="jobProgressStatus"
        :stroke-width="10"
        style="margin-top:10px"
      />
      <p class="job-message">{{ tr(job.message) || $t('arena.waitingSchedule') }}</p>
      <el-alert
        v-if="job.status === 'failed'"
        type="error"
        :closable="false"
        :title="job.error || $t('arena.jobFailed')"
      />
      <el-alert
        v-if="job.status === 'cancelled'"
        type="info"
        :closable="false"
        :title="$t('arena.jobCancelledNote')"
      />
    </el-card>

    <template v-if="result">
      <el-alert type="success" :closable="false" style="margin-bottom:16px">
        <template #title>
          {{ tr(result.dataset.name) }} · {{ $t('arena.splitsUnit', { count: result.n_splits }) }} · {{ $t('arena.dataSignature') }} {{ result.dataset.data_signature.slice(0, 16) }} ·
          {{ formatDuration(result.duration_ms) }}
          <template v-if="result.experiment_id">
            · {{ $t('arena.savedToDb') }} <b>#{{ result.experiment_id }}</b>
          </template>
        </template>
        <div class="result-actions">
          <el-button v-if="result.experiment_id" size="small" type="primary" link @click="openReport">{{ $t('arena.viewReport') }}</el-button>
          <el-button v-if="result.experiment_id" size="small" type="primary" link @click="downloadReport('html')">{{ $t('arena.downloadHtml') }}</el-button>
          <el-button v-if="result.experiment_id" size="small" type="primary" link @click="downloadReport('xlsx')">{{ $t('arena.downloadExcel') }}</el-button>
          <span v-if="result.cache" class="text-muted cache-hint">
            {{ $t('arena.cacheStats', { hits: result.cache.hits, misses: result.cache.misses, writes: result.cache.writes, entries: result.cache.entries, size: formatBytes(result.cache.bytes) }) }}
          </span>
        </div>
      </el-alert>

      <el-card shadow="never" style="margin-bottom:16px">
        <template #header>
          <div class="section-head">
            <b>{{ $t('arena.statistics') }}</b>
            <span class="text-muted">
              {{ statistics?.unit === 'subject' ? $t('arena.subjectLevel') : $t('arena.foldLevel') }} ·
              n = {{ statistics?.n_units ?? '-' }} · α = {{ statistics?.alpha ?? 0.05 }}
            </span>
          </div>
        </template>
        <el-alert type="info" :closable="false" :title="statistics?.note || $t('arena.noStatistics')" />
        <el-row :gutter="16" style="margin-top:12px">
          <el-col :xs="24" :md="12" style="margin-bottom:12px">
            <div class="section-title">{{ $t('arena.rankTitle') }}</div>
            <el-table :data="rankRows" border size="small">
              <el-table-column :label="$t('arena.rank')" width="70" align="center">
                <template #default="{ row, $index }">{{ $index + 1 }}</template>
              </el-table-column>
              <el-table-column :label="$t('arena.pipeline')" min-width="150">
                <template #default="{ row }">{{ pipelineName(row.pipeline) }}</template>
              </el-table-column>
              <el-table-column :label="$t('arena.meanRank')" width="90" align="center">
                <template #default="{ row }">{{ formatMetric(row.mean_rank, 2) }}</template>
              </el-table-column>
              <el-table-column :label="$t('arena.meanBA')" width="90" align="center">
                <template #default="{ row }">{{ formatMetric(row.mean_balanced_accuracy) }}</template>
              </el-table-column>
              <template #empty>
                <span class="text-muted">{{ $t('arena.rankEmpty') }}</span>
              </template>
            </el-table>
            <div class="stat-note">
              <template v-if="statistics?.friedman?.available">
                {{ $t('arena.friedmanSummary', {
                  statistic: formatMetric(statistics.friedman.statistic, 3),
                  p: formatP(statistics.friedman.p_value),
                  conclusion: statistics.friedman.significant ? $t('arena.significantDifference') : $t('arena.noSignificantDifference'),
                }) }}
              </template>
              <template v-else>{{ statistics?.friedman?.reason ? $t('arena.friedmanReason', { reason: statistics.friedman.reason }) : $t('arena.friedmanNeedsThree') }}</template>
              <template v-if="statistics?.nemenyi?.available">
                {{ $t('arena.nemenyiNote', { cd: formatMetric(statistics.nemenyi.critical_difference, 3) }) }}
              </template>
              <template v-else-if="statistics?.nemenyi?.reason"> {{ $t('arena.nemenyiReason', { reason: statistics.nemenyi.reason }) }}</template>
            </div>
          </el-col>
          <el-col :xs="24" :md="12" style="margin-bottom:12px">
            <div class="section-title">{{ $t('arena.pairwiseTitle') }}</div>
            <el-table :data="pairwiseRows" border size="small" max-height="320">
              <el-table-column :label="$t('arena.comparison')" min-width="180">
                <template #default="{ row }">
                  {{ pipelineName(row.pipeline_a) }} <span class="text-muted">vs</span> {{ pipelineName(row.pipeline_b) }}
                </template>
              </el-table-column>
              <el-table-column label="ΔBA" width="90" align="center">
                <template #default="{ row }">{{ formatSigned(row.mean_diff) }}</template>
              </el-table-column>
              <el-table-column label="p (Holm)" width="100" align="center">
                <template #default="{ row }">{{ formatP(row.p_value_adjusted) }}</template>
              </el-table-column>
              <el-table-column :label="$t('arena.significant')" width="70" align="center">
                <template #default="{ row }">
                  <el-tag size="small" :type="row.significant ? 'danger' : 'info'">{{ row.significant ? $t('arena.yes') : $t('arena.no') }}</el-tag>
                </template>
              </el-table-column>
              <template #empty>
                <span class="text-muted">{{ $t('arena.pairwiseEmpty') }}</span>
              </template>
            </el-table>
          </el-col>
        </el-row>
      </el-card>

      <el-card v-if="calibrationRows.length" shadow="never" style="margin-bottom:16px">
        <template #header>
          <div class="section-head">
            <b>{{ $t('arena.calibration') }}</b>
            <span class="text-muted">{{ $t('arena.calibrationHint') }}</span>
          </div>
        </template>
        <el-table :data="calibrationRows" border stripe size="small">
          <el-table-column :label="$t('arena.pipeline')" min-width="190">
            <template #default="{ row }">{{ pipelineName(row.pipeline) }}</template>
          </el-table-column>
          <el-table-column label="ECE" width="90" align="center">
            <template #default="{ row }">{{ formatMetric(row.ece) }}</template>
          </el-table-column>
          <el-table-column label="MCE" width="90" align="center">
            <template #default="{ row }">{{ formatMetric(row.mce) }}</template>
          </el-table-column>
          <el-table-column label="Brier" width="90" align="center">
            <template #default="{ row }">{{ formatMetric(row.brier) }}</template>
          </el-table-column>
          <el-table-column label="NLL" width="90" align="center">
            <template #default="{ row }">{{ formatMetric(row.nll) }}</template>
          </el-table-column>
          <el-table-column :label="$t('arena.confidenceAccuracy')" width="160" align="center">
            <template #default="{ row }">
              {{ formatMetric(row.confidence_mean, 3) }} / {{ formatMetric(row.accuracy, 3) }}
            </template>
          </el-table-column>
          <el-table-column prop="n_evaluated" :label="$t('arena.evaluatedTrials')" width="100" align="center" />
        </el-table>
      </el-card>

      <el-row :gutter="16">
        <el-col :xs="24" :md="24" :lg="15" class="arena-col">
          <el-card shadow="never">
            <template #header><b>{{ $t('arena.rankingTitle') }}</b></template>
            <el-table :data="result.rows" border stripe>
              <el-table-column type="expand">
                <template #default="{ row }">
                  <el-table :data="row.splits" size="small" border style="margin:8px 16px">
                    <el-table-column prop="split.fold" :label="$t('arena.fold')" width="60" align="center" />
                    <el-table-column prop="split.seed" :label="$t('arena.seed')" width="80" align="center" />
                    <el-table-column label="BA" width="100" align="center">
                      <template #default="{ row: split }">{{ formatMetric(split.balanced_accuracy) }}</template>
                    </el-table-column>
                    <el-table-column label="Kappa" width="100" align="center">
                      <template #default="{ row: split }">{{ formatMetric(split.cohen_kappa) }}</template>
                    </el-table-column>
                    <el-table-column :label="$t('arena.duration')" width="100" align="center">
                      <template #default="{ row: split }">{{ formatDuration(split.duration_ms) }}</template>
                    </el-table-column>
                    <el-table-column prop="split.signature" :label="$t('arena.splitSignature')" align="center" />
                  </el-table>
                </template>
              </el-table-column>
              <el-table-column type="index" label="#" width="50" align="center" />
              <el-table-column :label="$t('arena.pipeline')" min-width="200">
                <template #default="{ row }">
                  <b>{{ pipelineName(row.pipeline) }}</b>
                  <div class="text-muted mono" style="font-size:12px">{{ row.pipeline }}</div>
                </template>
              </el-table-column>
              <el-table-column :label="$t('arena.baMeanSd')" min-width="170" align="center">
                <template #default="{ row }">
                  <b>{{ formatMetric(row.balanced_accuracy_mean) }}</b>
                  <span class="text-muted"> ± {{ formatMetric(row.balanced_accuracy_std) }}</span>
                </template>
              </el-table-column>
              <el-table-column label="95% CI" min-width="140" align="center">
                <template #default="{ row }">{{ formatCi(row.balanced_accuracy_ci95) }}</template>
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
              <el-table-column :label="$t('arena.meanDuration')" width="110" align="center">
                <template #default="{ row }">{{ formatDuration(meanDuration(row)) }}</template>
              </el-table-column>
              <el-table-column prop="n_splits" :label="$t('arena.splitCount')" width="90" align="center" />
            </el-table>
          </el-card>
        </el-col>
        <el-col :xs="24" :md="24" :lg="9" class="arena-col">
          <el-card shadow="never">
            <template #header><b>{{ $t('arena.metricComparison') }}</b></template>
            <div ref="chartEl" style="height:340px; width:100%" />
          </el-card>
        </el-col>
      </el-row>
      <el-alert style="margin-top:16px" type="info" :closable="false" :title="result.note" />
    </template>

    <el-drawer v-model="jobsVisible" :title="$t('arena.jobQueue')" size="min(820px, 94%)">
      <div class="filter-bar">
        <el-button icon="Refresh" :loading="jobsLoading" @click="loadJobs">{{ $t('common.refresh') }}</el-button>
        <span class="text-muted" style="font-size:12px">{{ $t('arena.jobQueueHint') }}</span>
      </div>
      <el-table :data="jobsList" border stripe size="small" v-loading="jobsLoading">
        <el-table-column :label="$t('arena.task')" width="100">
          <template #default="{ row }"><span class="mono">{{ row.id.slice(0, 8) }}</span></template>
        </el-table-column>
        <el-table-column :label="$t('arena.status')" width="90" align="center">
          <template #default="{ row }">
            <el-tag size="small" :type="statusMeta(row.status).type">{{ statusMeta(row.status).label }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="$t('arena.progress')" width="140">
          <template #default="{ row }">
            <el-progress :percentage="Math.round((row.progress || 0) * 100)" :stroke-width="8" />
          </template>
        </el-table-column>
        <el-table-column :label="$t('arena.info')" min-width="200">
          <template #default="{ row }">
            <div>{{ tr(row.message) || '-' }}</div>
            <div v-if="row.error" class="text-muted" style="font-size:11px">{{ row.error }}</div>
          </template>
        </el-table-column>
        <el-table-column :label="$t('arena.submittedAt')" width="170" align="center">
          <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
        </el-table-column>
        <el-table-column :label="$t('arena.actions')" width="130" align="center" fixed="right">
          <template #default="{ row }">
            <el-button v-if="row.status === 'completed'" link type="primary" @click="viewJob(row)">{{ $t('arena.view') }}</el-button>
            <el-button v-if="['queued', 'running'].includes(row.status)" link type="danger" @click="cancelJobRow(row)">{{ $t('common.cancel') }}</el-button>
          </template>
        </el-table-column>
        <template #empty>
          <el-empty :description="$t('arena.emptyJobs')" />
        </template>
      </el-table>
    </el-drawer>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { tr } from '@/i18n/catalog'
import { ElMessage } from 'element-plus'
import {
  cancelJob,
  getCustomPipelines,
  getDatasets,
  getJob,
  getJobs,
  getPipelines,
  runArena,
  submitArenaJob,
} from '@/api'
import { createChart } from '@/utils/chart'
import { downloadWithAuth } from '@/utils/download'
import { formatDuration, formatMetric, formatTime } from '@/utils/format'
import { selection, setDataset, subjectChoices } from '@/utils/selection'

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

const JOB_STATUS = {
  queued: { labelKey: 'arena.jobStatusQueued', type: 'info' },
  running: { labelKey: 'arena.jobStatusRunning', type: 'warning' },
  completed: { labelKey: 'arena.jobStatusCompleted', type: 'success' },
  failed: { labelKey: 'arena.jobStatusFailed', type: 'danger' },
  cancelled: { labelKey: 'arena.jobStatusCancelled', type: 'info' },
}

const router = useRouter()
const route = useRoute()
const datasets = ref([])
const pipelines = ref([])
const result = ref(null)
const submitting = ref(false)
const seedText = ref('0')
const chartEl = ref(null)
const job = ref(null)
const jobsVisible = ref(false)
const jobsList = ref([])
const jobsLoading = ref(false)
let chartCtl = null
let pollTimer = null
let pollToken = 0

const form = reactive({
  protocol: 'within_subject', folds: 5, repeats: 1, classes: 2,
  trialsPerClass: 40, sessions: 2, pipelines: [], name: '', save: true, async: true,
  useCache: true,
})

const current = computed(() => datasets.value.find((item) => item.id === selection.datasetId))
const currentParadigm = computed(() => current.value?.paradigm || 'MI')
const compatiblePipelines = computed(() => pipelines.value.filter((item) => supportsParadigm(item, currentParadigm.value)))
const calibrationRows = computed(() => (
  (result.value?.rows || [])
    .map((row) => ({ pipeline: row.pipeline, ...(row.calibration || {}) }))
    .filter((row) => row.available)
))
const subjectOptions = computed(() => subjectChoices(current.value))
const classOptions = computed(() => {
  const count = current.value?.classes || 2
  return count > 2 ? Array.from({ length: count - 1 }, (_, index) => index + 2) : []
})
const isSynthetic = computed(() => current.value?.adapter === 'builtin')
const hasDeep = computed(() => form.pipelines.some((id) => ['eegnet', 'shallowconvnet'].includes(id)))
const statistics = computed(() => result.value?.statistics)
const rankRows = computed(() => statistics.value?.average_ranks?.ranks || [])
const pairwiseRows = computed(() => statistics.value?.pairwise || [])
const jobActive = computed(() => !!job.value && ['queued', 'running'].includes(job.value.status))
const jobProgress = computed(() => Math.round((job.value?.progress || 0) * 100))
const jobStatusType = computed(() => statusMeta(job.value?.status).type)
const jobStatusLabel = computed(() => statusMeta(job.value?.status).label)
const jobProgressStatus = computed(() => {
  if (job.value?.status === 'completed') return 'success'
  if (job.value?.status === 'failed') return 'exception'
  if (job.value?.status === 'cancelled') return 'warning'
  return ''
})

function statusMeta(status) {
  const meta = JOB_STATUS[status]
  if (!meta) return { label: status || '-', type: 'info' }
  return { label: t(meta.labelKey), type: meta.type }
}

function pipelineName(id) {
  return tr(pipelines.value.find((item) => item.id === id)?.name) || id || '-'
}

function shortLabel(id) {
  if (SHORT_LABELS[id]) return SHORT_LABELS[id]
  const name = pipelineName(id)
  return name.length > 9 ? `${name.slice(0, 9)}…` : name
}

function meanDuration(row) {
  const values = (row.splits || []).map((split) => split.duration_ms).filter((value) => value !== undefined)
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null
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

function formatCi(value) {
  if (!Array.isArray(value) || value.length !== 2) return '-'
  if (value.some((item) => item === null || item === undefined)) return '-'
  return `[${Number(value[0]).toFixed(3)}, ${Number(value[1]).toFixed(3)}]`
}

function supportsParadigm(item, paradigm) {
  const label = item?.paradigm || 'MI'
  const supported = label.split('/').map((part) => part.trim().toUpperCase()).filter(Boolean)
  return supported.includes(String(paradigm || 'MI').toUpperCase())
}

function defaultPipelines(paradigm) {
  if (paradigm === 'P300') return ['xdawn_lda', 'raw_centroid']
  if (paradigm === 'SSVEP') return ['ssvep_fbcca', 'ssvep_cca', 'raw_centroid']
  return ['csp_lda', 'riemann_mdm', 'raw_centroid']
}

function applyPipelineDefaults(force = false) {
  const allowed = new Set(compatiblePipelines.value.map((item) => item.id))
  const kept = form.pipelines.filter((id) => allowed.has(id))
  form.pipelines = force || !kept.length
    ? defaultPipelines(currentParadigm.value).filter((id) => allowed.has(id))
    : kept
}

function formatBytes(value) {
  if (value === null || value === undefined) return '-'
  const units = ['B', 'KiB', 'MiB', 'GiB']
  let size = Number(value)
  let index = 0
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024
    index += 1
  }
  return `${size.toFixed(index === 0 ? 0 : 1)} ${units[index]}`
}

function syncDefaults() {
  const dataset = current.value
  if (!dataset) return
  form.classes = dataset.classes || 2
  const options = subjectChoices(dataset)
  const valid = selection.subjectIds.filter((id) => options.includes(id))
  selection.subjectIds = valid.length ? valid : options.slice(0, 1)
  applyPipelineDefaults()
}

function onDatasetChange(datasetId) {
  setDataset(datasetId)
  result.value = null
  disposeChart()
  applyPipelineDefaults(true)
  syncDefaults()
}

function buildPayload() {
  const options = {}
  if (isSynthetic.value) {
    options.n_trials_per_class = form.trialsPerClass
    if (form.protocol === 'cross_session') options.n_sessions = Math.max(2, form.sessions)
  }
  const seeds = seedText.value
    .split(',')
    .map((item) => parseInt(item.trim(), 10))
    .filter((value) => !Number.isNaN(value))
  return {
    dataset_id: selection.datasetId,
    subject_ids: selection.subjectIds,
    classes: form.classes,
    options,
    protocol: form.protocol,
    folds: form.folds,
    repeats: form.repeats,
    seeds: seeds.length ? seeds : [0],
    pipelines: form.pipelines,
    name: form.name,
    save: form.save,
    use_cache: form.useCache,
  }
}

function validate() {
  if (!form.pipelines.length) {
    ElMessage.warning(t('arena.selectPipelineRequired'))
    return false
  }
  if (form.protocol === 'loso' && selection.subjectIds.length < 2) {
    ElMessage.warning(t('arena.losoNeedsTwoSubjects'))
    return false
  }
  if (form.protocol !== 'loso' && selection.subjectIds.length !== 1) {
    ElMessage.warning(t('arena.singleSubjectOnly'))
    return false
  }
  return true
}

function disposeChart() {
  chartCtl?.dispose()
  chartCtl = null
}

function renderChart() {
  if (!chartEl.value || !result.value) return
  if (!chartCtl) chartCtl = createChart(chartEl.value)
  const rows = result.value.rows
  chartCtl.chart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { top: 0, data: ['Balanced Accuracy', 'Cohen Kappa'] },
    grid: { left: 12, right: 16, top: 44, bottom: 16, containLabel: true },
    xAxis: {
      type: 'category',
      data: rows.map((row) => shortLabel(row.pipeline)),
      axisLabel: { interval: 0, fontSize: 11, color: '#606266' },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value', min: 0, max: 1,
      axisLabel: { color: '#909399' },
      splitLine: { lineStyle: { type: 'dashed' } },
    },
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

async function applyResult(payload) {
  if (!payload) return
  result.value = payload
  if (payload.experiment_id) ElMessage.success(t('arena.experimentSaved', { id: payload.experiment_id }))
  await nextTick()
  renderChart()
  requestAnimationFrame(() => chartCtl?.chart.resize())
}

async function run() {
  if (!validate()) return
  const payload = buildPayload()
  if (!form.async) {
    stopPolling()
    job.value = null
    submitting.value = true
    try {
      result.value = await runArena(payload)
      await applyResult(result.value)
    } finally {
      submitting.value = false
    }
    return
  }
  submitting.value = true
  try {
    result.value = null
    disposeChart()
    const submitted = await submitArenaJob(payload)
    job.value = submitted
    ElMessage.success(t('arena.jobSubmitted'))
    schedulePoll(400)
  } finally {
    submitting.value = false
  }
}

function schedulePoll(delay = 1000) {
  clearTimeout(pollTimer)
  const token = ++pollToken
  pollTimer = setTimeout(() => pollJob(token), delay)
}

function stopPolling() {
  pollToken += 1
  clearTimeout(pollTimer)
  pollTimer = null
}

async function pollJob(token) {
  if (!job.value || token !== pollToken) return
  try {
    const latest = await getJob(job.value.id)
    if (token !== pollToken) return
    job.value = latest
    if (['queued', 'running'].includes(latest.status)) {
      schedulePoll(1000)
      return
    }
    if (latest.status === 'completed') {
      await applyResult(latest.result)
    } else if (latest.status === 'failed') {
      ElMessage.error(latest.error || t('arena.backgroundJobFailed'))
    }
  } catch {
    // the response interceptor already reported the error; stop this poll cycle
  }
}

async function cancelActiveJob() {
  if (!job.value) return
  try {
    await cancelJob(job.value.id)
    job.value = { ...job.value, cancel_requested: true, message: t('arena.cancelRequestedMessage') }
    ElMessage.info(t('arena.cancelRequested'))
    schedulePoll(600)
  } catch {
    // interceptor already reported the reason
  }
}

async function loadJobs() {
  jobsLoading.value = true
  try {
    const response = await getJobs({ limit: 50 })
    jobsList.value = response.items.filter((item) => item.kind === 'arena')
  } finally {
    jobsLoading.value = false
  }
}

async function openJobs() {
  jobsVisible.value = true
  await loadJobs()
}

async function viewJob(row) {
  const latest = await getJob(row.id)
  if (latest.status === 'completed') {
    job.value = latest
    await applyResult(latest.result)
    jobsVisible.value = false
  } else if (['queued', 'running'].includes(latest.status)) {
    job.value = latest
    jobsVisible.value = false
    schedulePoll(600)
  } else {
    ElMessage.info(t('arena.noJobResult'))
  }
}

async function cancelJobRow(row) {
  await cancelJob(row.id)
  ElMessage.info(t('arena.cancelRequested'))
  await loadJobs()
}

async function downloadReport(format) {
  const id = result.value?.experiment_id
  if (!id) return
  try {
    await downloadWithAuth(
      `/api/experiments/${id}/report.${format}`,
      `neurokiln-experiment-${id}.${format}`,
    )
    ElMessage.success(t('arena.reportDownloading'))
  } catch (error) {
    ElMessage.error(error.message)
  }
}

function openReport() {
  router.push({ path: '/report', query: { experiment: result.value.experiment_id } })
}

onMounted(async () => {
  const [datasetResponse, pipelineResponse, customResponse] = await Promise.all([
    getDatasets(),
    getPipelines(),
    getCustomPipelines(),
  ])
  datasets.value = datasetResponse.items
  pipelines.value = [
    ...pipelineResponse.items,
    ...customResponse.items.map((item) => ({
      id: `custom:${item.id}`,
      name: item.name,
      family: 'custom',
      paradigm: item.paradigm || 'MI',
      note: item.description,
    })),
  ]
  if (!datasets.value.some((item) => item.id === selection.datasetId)) {
    setDataset(datasets.value[0]?.id || 'synthetic')
  }
  applyPipelineDefaults(true)
  const requested = route.query.pipeline
  if (typeof requested === 'string' && compatiblePipelines.value.some((item) => item.id === requested)) {
    form.pipelines = [requested]
  }
  syncDefaults()
})

onBeforeUnmount(() => {
  stopPolling()
  disposeChart()
})
</script>

<style scoped>
.arena-col { margin-bottom: 16px; }
.job-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; }
.job-title { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.job-pipelines { font-size: 12px; }
.job-actions { display: flex; gap: 8px; }
.job-message { margin: 8px 0 10px; color: #606266; font-size: 13px; }
.result-actions { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; align-items: center; }
.cache-hint { font-size: 12px; margin-left: 10px; }
.section-head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; }
.section-title { font-size: 13px; font-weight: 600; margin-bottom: 8px; }
.stat-note { font-size: 12px; line-height: 1.8; color: #606266; margin-top: 8px; }
</style>
