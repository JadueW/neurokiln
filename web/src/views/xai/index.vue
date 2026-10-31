<template>
  <div class="app-container">
    <h2 class="page-title">{{ $t('xai.title') }}</h2>
    <p class="page-desc">{{ $t('xai.description') }}</p>

    <el-card shadow="never" style="margin-bottom:16px">
      <el-form :inline="true" label-width="150px">
        <el-form-item :label="$t('xai.dataset')">
          <el-select :model-value="selection.datasetId" style="width:220px" @change="onDatasetChange">
            <el-option v-for="item in datasets" :key="item.id" :label="tr(item.name)" :value="item.id" :disabled="!item.ready">
              <span>{{ tr(item.name) }}</span>
              <el-tag size="small" type="info" style="float:right;margin-left:12px">{{ item.paradigm }}</el-tag>
            </el-option>
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('xai.subject')">
          <el-select v-model="subjectId" style="width:120px" :placeholder="$t('arena.defaultSubjects')">
            <el-option v-for="subject in subjectOptions" :key="subject" :label="subject" :value="subject" />
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('xai.pipeline')">
          <el-select v-model="form.pipeline" style="width:230px">
            <el-option v-for="item in compatiblePipelines" :key="item.id" :label="tr(item.name)" :value="item.id">
              <span>{{ tr(item.name) }}</span>
              <el-tag size="small" type="info" style="float:right;margin-left:12px">{{ item.paradigm || 'MI' }}</el-tag>
            </el-option>
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('arena.folds')"><el-input-number v-model="form.folds" :min="2" :max="10" /></el-form-item>
        <el-form-item :label="$t('arena.repeats')"><el-input-number v-model="form.repeats" :min="1" :max="5" /></el-form-item>
        <el-form-item :label="$t('xai.fold')"><el-input-number v-model="form.fold" :min="0" :max="form.folds - 1" /></el-form-item>
        <el-form-item :label="$t('xai.repeat')"><el-input-number v-model="form.repeat" :min="0" :max="form.repeats - 1" /></el-form-item>
        <el-form-item :label="$t('xai.seed')"><el-input v-model="form.seedText" style="width:100px" /></el-form-item>
        <el-form-item :label="$t('xai.gradientMethod')">
          <el-select v-model="form.gradientMethod" style="width:180px">
            <el-option :label="$t('xai.gradientIg')" value="integrated_gradients" />
            <el-option :label="$t('xai.gradientDeeplift')" value="deeplift" />
            <el-option :label="$t('xai.gradientNone')" value="none" />
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('xai.windowMs')"><el-input-number v-model="form.windowMs" :min="50" :max="2000" :step="50" /></el-form-item>
        <el-form-item :label="$t('xai.maxBlocks')"><el-input-number v-model="form.maxBlocks" :min="8" :max="600" :step="8" /></el-form-item>
        <el-form-item :label="$t('xai.experiment')">
          <el-select v-model="form.experimentId" clearable style="width:260px" :placeholder="$t('xai.noExperiment')" @change="onExperimentChange">
            <el-option
              v-for="item in experiments"
              :key="item.id"
              :label="`#${item.id} · ${tr(item.name)}`"
              :value="item.id"
            />
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('xai.includeAudit')"><el-switch v-model="form.audit" /></el-form-item>
        <el-form-item :label="$t('xai.includeSanity')"><el-switch v-model="form.sanity" /></el-form-item>
        <el-form-item>
          <el-button type="primary" icon="VideoPlay" :loading="submitting" :disabled="jobActive" @click="run">{{ $t('xai.run') }}</el-button>
          <el-button icon="Timer" @click="openHistory">{{ $t('xai.history') }}</el-button>
          <el-button icon="DataLine" :loading="aggregateLoading" @click="openAggregate">{{ $t('xai.aggregate') }}</el-button>
        </el-form-item>
      </el-form>
      <el-alert type="info" :closable="false" :title="$t('xai.modelOnlyHint')" style="margin-bottom:6px" />
      <el-alert type="info" :closable="false" :title="$t('xai.splitHint')" style="margin-bottom:6px" />
      <el-alert type="warning" :closable="false" :title="$t('xai.gradientHint')" />
    </el-card>

    <el-card v-if="job" shadow="never" style="margin-bottom:16px">
      <div class="job-row">
        <div class="job-title">
          <b>{{ $t('xai.jobTitle') }}</b>
          <span class="mono text-muted">{{ job.id.slice(0, 8) }}</span>
          <el-tag size="small" :type="jobStatusType">{{ jobStatusLabel }}</el-tag>
          <span v-if="job.pipeline_name" class="text-muted job-pipeline">{{ tr(job.pipeline_name) }}</span>
        </div>
        <div class="job-actions">
          <el-button v-if="jobActive" size="small" type="danger" plain icon="Close" @click="cancelActiveJob">{{ $t('xai.cancelJob') }}</el-button>
          <el-button v-if="job.status === 'completed' && job.result" size="small" type="primary" @click="openExplanation(job.result.explanation_id)">{{ $t('xai.viewResult') }}</el-button>
          <el-button v-if="!jobActive" size="small" @click="job = null">{{ $t('xai.collapse') }}</el-button>
        </div>
      </div>
      <el-progress :percentage="jobProgress" :status="jobProgressStatus" :stroke-width="10" style="margin-top:10px" />
      <p class="job-message">{{ tr(job.message) || $t('xai.waiting') }}</p>
      <el-alert v-if="job.status === 'failed'" type="error" :closable="false" :title="jobErrorMessage(job, $t('xai.jobFailed'))" />
      <el-alert v-if="job.status === 'cancelled'" type="info" :closable="false" :title="$t('xai.jobCancelledNote')" />
    </el-card>

    <template v-if="doc">
      <el-alert type="success" :closable="false" style="margin-bottom:16px">
        <template #title>
          #{{ explanation.id }} · {{ tr(explanation.pipeline_name) }} ·
          {{ $t('xai.splitLine', {
            protocol: doc.split?.protocol,
            fold: doc.split?.fold,
            repeat: doc.split?.repeat,
            seed: doc.split?.seed,
            n: doc.performance?.n_test,
          }) }}
          <span class="text-muted"> · {{ formatDuration(explanation.duration_ms) }}</span>
        </template>
        <div class="result-actions">
          <el-button size="small" type="primary" link @click="exportDocument">{{ $t('xai.exportJson') }}</el-button>
          <el-button v-if="explanation.experiment_id" size="small" type="primary" link @click="openReport">
            {{ $t('arena.viewReport') }} #{{ explanation.experiment_id }}
          </el-button>
          <span class="text-muted result-note">{{ $t('xai.scoreUnit', { unit: doc.occlusion?.score_unit || doc.faithfulness?.score_unit || '-' }) }}</span>
        </div>
      </el-alert>

      <div class="stat-grid" style="margin-bottom:16px">
        <el-card v-for="card in metricCards" :key="card.label" shadow="never">
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

      <el-alert
        v-for="(warning, index) in doc.warnings || []"
        :key="index"
        type="warning"
        :closable="false"
        :title="tr(warning)"
        style="margin-bottom:8px"
      />

      <el-card shadow="never" style="margin-bottom:16px">
        <template #header>
          <div class="section-head">
            <b>{{ $t('xai.figures') }}</b>
            <span class="text-muted">{{ $t('xai.signedLegend') }}</span>
          </div>
        </template>
        <div v-if="figures.length" class="figure-grid">
          <figure v-for="figure in figures" :key="figure.key" class="xai-fig">
            <img :src="figure.src" :alt="figure.caption" />
            <figcaption>{{ figure.caption }}</figcaption>
          </figure>
        </div>
        <el-empty v-else :description="$t('xai.noFigures')" :image-size="72" />
      </el-card>

      <el-row :gutter="16" style="margin-bottom:16px">
        <el-col :xs="24" :md="12" class="panel-col">
          <el-card shadow="never">
            <template #header><b>{{ $t('xai.curves') }}</b></template>
            <div v-if="curvesAvailable" ref="curvesEl" style="height:320px;width:100%" />
            <el-alert v-else type="warning" :closable="false" :title="curveWarning" />
            <p class="stat-note">{{ $t('xai.curvesHint') }}</p>
          </el-card>
        </el-col>
        <el-col :xs="24" :md="12" class="panel-col">
          <el-card shadow="never">
            <template #header><b>{{ $t('xai.bands') }}</b></template>
            <div v-if="bandsAvailable" ref="bandsEl" style="height:320px;width:100%" />
            <el-empty v-else :description="bandsReason" :image-size="72" />
          </el-card>
        </el-col>
      </el-row>

      <el-card v-if="heatmapData" shadow="never" style="margin-bottom:16px">
        <template #header>
          <div class="section-head">
            <b>{{ $t('xai.occlusionColumn') }} · {{ heatmapWindowLabel }}</b>
            <span class="text-muted">{{ $t('xai.signedLegend') }}</span>
          </div>
        </template>
        <div ref="heatEl" :style="{ height: heatmapHeight, width: '100%' }" />
      </el-card>

      <el-card shadow="never" style="margin-bottom:16px">
        <template #header><b>{{ $t('xai.channels') }}</b></template>
        <el-table v-if="channelRows.length" :data="channelRows" border stripe size="small" max-height="420">
          <el-table-column prop="name" :label="$t('xai.channel')" min-width="110" fixed />
          <el-table-column :label="$t('xai.occlusionColumn')" width="110" align="center">
            <template #default="{ row }">{{ formatValue(row.occlusion) }}</template>
          </el-table-column>
          <el-table-column :label="$t('xai.intrinsicColumn')" width="110" align="center">
            <template #default="{ row }">{{ formatValue(row.intrinsic) }}</template>
          </el-table-column>
          <el-table-column :label="$t('xai.bandColumn')" width="110" align="center">
            <template #default="{ row }">{{ formatValue(row.band) }}</template>
          </el-table-column>
          <el-table-column :label="$t('xai.gradientColumn')" width="110" align="center">
            <template #default="{ row }">{{ formatValue(row.gradient) }}</template>
          </el-table-column>
          <el-table-column :label="$t('xai.bandPowerColumn')" width="120" align="center">
            <template #default="{ row }">{{ formatValue(row.power) }}</template>
          </el-table-column>
        </el-table>
        <el-empty v-else :description="$t('xai.channelsEmpty')" :image-size="72" />
      </el-card>

      <el-row :gutter="16" style="margin-bottom:16px">
        <el-col :xs="24" :md="12" class="panel-col">
          <el-card shadow="never">
            <template #header><b>{{ $t('xai.audit') }}</b></template>
            <template v-if="auditAvailable">
              <el-descriptions :column="1" border size="small">
                <el-descriptions-item :label="$t('xai.closure')">{{ formatMetric(doc.audit.closure_ratio) }}</el-descriptions-item>
                <el-descriptions-item :label="$t('xai.student')">
                  {{ $t('xai.studentLine', {
                    model: doc.audit.student?.model,
                    ba: formatMetric(doc.audit.student?.balanced_accuracy),
                  }) }}
                </el-descriptions-item>
                <el-descriptions-item :label="$t('xai.surrogate')">
                  {{ $t('xai.surrogateLine', {
                    model: doc.audit.surrogate?.model,
                    r2: formatMetric(doc.audit.surrogate?.r2),
                  }) }}
                </el-descriptions-item>
              </el-descriptions>
              <div class="section-title" style="margin-top:12px">{{ $t('xai.topFeatures') }}</div>
              <el-table :data="doc.audit.top_features || []" border size="small" max-height="240">
                <el-table-column prop="name" :label="$t('xai.feature')" min-width="180" />
                <el-table-column :label="$t('xai.weight')" width="100" align="center">
                  <template #default="{ row }">{{ formatValue(row.weight) }}</template>
                </el-table-column>
              </el-table>
            </template>
            <el-empty v-else :description="$t('xai.auditUnavailable', { reason: tr(doc.audit?.reason) || '-' })" :image-size="72" />
          </el-card>
        </el-col>
        <el-col :xs="24" :md="12" class="panel-col">
          <el-card shadow="never">
            <template #header><b>{{ $t('xai.provenance') }}</b></template>
            <el-descriptions :column="1" border size="small">
              <el-descriptions-item :label="$t('xai.sanity')">
                {{ sanityLine }}
              </el-descriptions-item>
              <template v-for="line in agreementLines" :key="line.label">
                <el-descriptions-item :label="line.label">{{ line.value }}</el-descriptions-item>
              </template>
              <el-descriptions-item :label="$t('xai.provenance')">{{ provenanceLine }}</el-descriptions-item>
            </el-descriptions>
            <div class="section-title" style="margin-top:12px">{{ $t('xai.notes') }}</div>
            <ul class="note-list">
              <li v-for="(note, index) in doc.provenance?.notes || []" :key="index">{{ tr(note) }}</li>
            </ul>
            <p v-if="doc.audit?.interpretation" class="stat-note">{{ tr(doc.audit.interpretation) }}</p>
            <p v-if="doc.faithfulness?.interpretation" class="stat-note">{{ tr(doc.faithfulness.interpretation) }}</p>
          </el-card>
        </el-col>
      </el-row>
    </template>

    <el-drawer v-model="historyVisible" :title="$t('xai.historyTitle')" size="min(980px, 94%)">
      <div class="filter-bar">
        <el-select v-model="historyExperiment" clearable style="width:260px" :placeholder="$t('xai.allExperiments')" @change="loadHistory">
          <el-option
            v-for="item in experiments"
            :key="item.id"
            :label="`#${item.id} · ${tr(item.name)}`"
            :value="item.id"
          />
        </el-select>
        <el-button icon="Refresh" :loading="historyLoading" @click="loadHistory">{{ $t('common.refresh') }}</el-button>
        <span class="text-muted" style="font-size:12px">{{ $t('xai.historyHint') }}</span>
      </div>
      <el-table :data="history" border stripe size="small" v-loading="historyLoading">
        <el-table-column prop="id" label="#" width="60" align="center" />
        <el-table-column :label="$t('xai.pipeline')" min-width="150">
          <template #default="{ row }">{{ tr(row.pipeline_name) }}</template>
        </el-table-column>
        <el-table-column :label="$t('xai.fold')" width="110" align="center">
          <template #default="{ row }">{{ row.fold }} / {{ row.repeat }} / {{ row.seed }}</template>
        </el-table-column>
        <el-table-column :label="$t('xai.experiment')" width="90" align="center">
          <template #default="{ row }">{{ row.experiment_id ? `#${row.experiment_id}` : '-' }}</template>
        </el-table-column>
        <el-table-column :label="$t('xai.status')" width="90" align="center">
          <template #default="{ row }">
            <el-tag size="small" :type="statusMeta(row.status).type">{{ statusMeta(row.status).label }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="$t('xai.metricBa')" width="90" align="center">
          <template #default="{ row }">{{ formatMetric(row.summary?.performance?.balanced_accuracy) }}</template>
        </el-table-column>
        <el-table-column :label="$t('xai.metricDeletion')" width="100" align="center">
          <template #default="{ row }">{{ formatMetric(row.summary?.deletion_auc) }}</template>
        </el-table-column>
        <el-table-column :label="$t('xai.createdAt')" width="165" align="center">
          <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
        </el-table-column>
        <el-table-column :label="$t('xai.actions')" width="130" align="center" fixed="right">
          <template #default="{ row }">
            <el-button v-if="row.status === 'completed'" link type="primary" @click="openExplanation(row.id)">{{ $t('xai.open') }}</el-button>
            <el-button link type="danger" @click="removeExplanation(row)">{{ $t('xai.delete') }}</el-button>
          </template>
        </el-table-column>
        <template #empty>
          <el-empty :description="$t('xai.historyEmpty')" />
        </template>
      </el-table>
    </el-drawer>

    <el-drawer v-model="aggregateVisible" :title="$t('xai.aggregateTitle')" size="min(1080px, 94%)">
      <div class="filter-bar">
        <el-select v-model="aggregateExperiment" clearable style="width:260px" :placeholder="$t('xai.aggregateAll')" @change="loadAggregate">
          <el-option
            v-for="item in experiments"
            :key="item.id"
            :label="`#${item.id} · ${tr(item.name)}`"
            :value="item.id"
          />
        </el-select>
        <el-button icon="Refresh" :loading="aggregateLoading" @click="loadAggregate">{{ $t('xai.aggregateLoad') }}</el-button>
        <span class="text-muted" style="font-size:12px">{{ $t('xai.aggregateFilterHint') }}</span>
      </div>
      <el-alert type="info" :closable="false" :title="$t('xai.aggregateHint')" style="margin-bottom:12px" />
      <div v-loading="aggregateLoading" :element-loading-text="$t('xai.aggregateLoading')">
        <template v-if="aggregate">
          <div class="stat-grid" style="margin-bottom:16px">
            <el-card v-for="card in aggregateCards" :key="card.label" shadow="never">
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

          <el-alert
            v-for="(warning, index) in aggregate.warnings || []"
            :key="`agg-${index}`"
            type="warning"
            :closable="false"
            :title="`${tr(warning.text)} × ${warning.count}`"
            style="margin-bottom:8px"
          />

          <el-card shadow="never" style="margin-bottom:16px">
            <template #header><b>{{ $t('xai.aggregateTopChannels') }}</b></template>
            <el-table :data="aggregate.top_channels || []" border stripe size="small" max-height="360">
              <el-table-column prop="name" :label="$t('xai.aggregateChannel')" min-width="110" fixed />
              <el-table-column :label="$t('xai.aggregateMean')" width="120" align="center">
                <template #default="{ row }">{{ formatValue(row.occlusion_mean) }}</template>
              </el-table-column>
              <el-table-column :label="$t('xai.aggregateStd')" width="120" align="center">
                <template #default="{ row }">{{ formatValue(row.occlusion_std) }}</template>
              </el-table-column>
              <el-table-column :label="$t('xai.aggregateSupport')" width="100" align="center">
                <template #default="{ row }">{{ row.support }}</template>
              </el-table-column>
              <el-table-column :label="$t('xai.intrinsicColumn')" width="120" align="center">
                <template #default="{ row }">{{ formatValue(row.intrinsic_mean) }}</template>
              </el-table-column>
            </el-table>
          </el-card>

          <el-card shadow="never" style="margin-bottom:16px">
            <template #header><b>{{ $t('xai.aggregateBands') }}</b></template>
            <el-table v-if="(aggregate.bands || []).length" :data="aggregate.bands" border stripe size="small">
              <el-table-column :label="$t('xai.aggregateBand')" min-width="120">
                <template #default="{ row }">{{ (row.band || []).join('–') }}</template>
              </el-table-column>
              <el-table-column :label="$t('xai.aggregateDrop')" min-width="190" align="center">
                <template #default="{ row }">{{ formatSummary(row.drop) }}</template>
              </el-table-column>
              <el-table-column :label="$t('xai.aggregateSupport')" width="110" align="center">
                <template #default="{ row }">{{ row.channel_importance?.n?.[0] ?? 0 }}</template>
              </el-table-column>
            </el-table>
            <el-empty v-else :description="$t('xai.bandsEmpty')" :image-size="64" />
          </el-card>

          <el-card shadow="never">
            <template #header><b>{{ $t('xai.aggregateSplitsTitle') }}</b></template>
            <el-table :data="aggregate.splits || []" border stripe size="small">
              <el-table-column :label="$t('xai.aggregateFold')" min-width="150">
                <template #default="{ row }">{{ row.fold }} / {{ row.repeat }} / {{ row.seed }}</template>
              </el-table-column>
              <el-table-column :label="$t('xai.aggregateBaColumn')" width="110" align="center">
                <template #default="{ row }">{{ formatMetric(row.balanced_accuracy) }}</template>
              </el-table-column>
              <el-table-column :label="$t('xai.aggregateDeletionColumn')" width="130" align="center">
                <template #default="{ row }">{{ formatMetric(row.deletion_auc) }}</template>
              </el-table-column>
              <el-table-column :label="$t('xai.aggregateWarningColumn')" width="110" align="center">
                <template #default="{ row }">{{ row.warnings }}</template>
              </el-table-column>
            </el-table>
            <div style="margin-top:12px">
              <el-button size="small" icon="Download" @click="exportAggregate">{{ $t('xai.aggregateExport') }}</el-button>
            </div>
          </el-card>
        </template>
        <el-empty
          v-else-if="!aggregateLoading"
          :description="aggregateError || $t('xai.aggregateEmpty')"
          :image-size="72"
        />
      </div>
    </el-drawer>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { tr } from '@/i18n/catalog'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  cancelJob,
  deleteExplanation,
  getDatasets,
  getExperiments,
  getExplanation,
  getExplanationAggregate,
  getExplanations,
  getJob,
  getPipelines,
  submitExplanationJob,
} from '@/api'
import { createChart } from '@/utils/chart'
import { downloadJson, formatDuration, formatMetric, formatTime } from '@/utils/format'
import { jobErrorMessage } from '@/utils/errors'
import { selection, setDataset, subjectChoices } from '@/utils/selection'

const { t } = useI18n()

const JOB_STATUS = {
  queued: { labelKey: 'arena.jobStatusQueued', type: 'info' },
  running: { labelKey: 'arena.jobStatusRunning', type: 'warning' },
  completed: { labelKey: 'arena.jobStatusCompleted', type: 'success' },
  failed: { labelKey: 'arena.jobStatusFailed', type: 'danger' },
  cancelled: { labelKey: 'arena.jobStatusCancelled', type: 'info' },
}

const EXPLANATION_STATUS = {
  running: { labelKey: 'xai.statusRunning', type: 'warning' },
  completed: { labelKey: 'xai.statusCompleted', type: 'success' },
  failed: { labelKey: 'xai.statusFailed', type: 'danger' },
}

const router = useRouter()
const route = useRoute()
const datasets = ref([])
const pipelines = ref([])
const experiments = ref([])
const subjectId = ref('')
const explanation = ref(null)
const doc = ref(null)
const images = ref({})
const submitting = ref(false)
const job = ref(null)
const historyVisible = ref(false)
const history = ref([])
const historyLoading = ref(false)
const historyExperiment = ref(null)
const aggregateVisible = ref(false)
const aggregate = ref(null)
const aggregateLoading = ref(false)
const aggregateExperiment = ref(null)
const aggregateError = ref('')
const curvesEl = ref(null)
const bandsEl = ref(null)
const heatEl = ref(null)
let curvesCtl = null
let bandsCtl = null
let heatCtl = null
let pollTimer = null
let pollToken = 0

const form = reactive({
  pipeline: 'csp_lda',
  fold: 0,
  repeat: 0,
  seedText: '0',
  folds: 5,
  repeats: 1,
  experimentId: null,
  gradientMethod: 'integrated_gradients',
  windowMs: 250,
  maxBlocks: 120,
  audit: true,
  sanity: true,
  trialsPerClass: 40,
})

const current = computed(() => datasets.value.find((item) => item.id === selection.datasetId))
const subjectOptions = computed(() => subjectChoices(current.value))
const isSynthetic = computed(() => current.value?.adapter === 'builtin')
const compatiblePipelines = computed(() => pipelines.value.filter(
  (item) => supportsParadigm(item, current.value?.paradigm || 'MI'),
))
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

const auditAvailable = computed(() => !!doc.value?.audit?.available)
const bandsAvailable = computed(() => !!doc.value?.bands?.available)
const bandsReason = computed(() => tr(doc.value?.bands?.reason) || t('xai.bandsEmpty'))
const curvesAvailable = computed(() => {
  const curve = doc.value?.faithfulness?.deletion_curve
  return !!doc.value?.faithfulness?.normalization?.usable
    && Array.isArray(curve?.fraction)
    && Array.isArray(curve?.retained)
})
const curveWarning = computed(() => tr(doc.value?.faithfulness?.normalization?.note) || t('xai.curvesHint'))

const metricCards = computed(() => {
  const current = doc.value
  if (!current) return []
  const faith = current.faithfulness || {}
  return [
    { label: t('xai.metricBa'), value: formatMetric(current.performance?.balanced_accuracy), icon: 'DataAnalysis', color: '#409eff' },
    { label: t('xai.metricKappa'), value: formatMetric(current.performance?.cohen_kappa), icon: 'Histogram', color: '#9254de' },
    { label: t('xai.metricDeletion'), value: formatMetric(faith.deletion_auc), icon: 'ArrowDown', color: '#f56c6c' },
    { label: t('xai.metricInsertion'), value: formatMetric(faith.insertion_auc), icon: 'TrendCharts', color: '#67c23a' },
    { label: t('xai.metricRandom'), value: formatMetric(faith.random_deletion_auc), icon: 'Rank', color: '#909399' },
    { label: t('xai.metricAopc'), value: formatMetric(faith.aopc), icon: 'Aim', color: '#e6a23c' },
    {
      label: t('xai.metricClosure'),
      value: formatMetric(current.audit?.closure_ratio),
      icon: 'Collection',
      color: '#00b8a9',
    },
  ]
})

const aggregateCards = computed(() => {
  const current = aggregate.value
  if (!current) return []
  return [
    {
      label: t('xai.aggregateN'),
      value: String(current.provenance?.n_explanations ?? 0),
      icon: 'Collection',
      color: '#409eff',
    },
    {
      label: t('xai.aggregateSplits'),
      value: String(current.provenance?.n_unique_splits ?? 0),
      icon: 'Files',
      color: '#9254de',
    },
    {
      label: t('xai.aggregateBa'),
      value: formatSummary(current.performance?.balanced_accuracy),
      icon: 'DataAnalysis',
      color: '#67c23a',
    },
    {
      label: t('xai.aggregateDeletion'),
      value: formatSummary(current.faithfulness?.deletion_auc),
      icon: 'ArrowDown',
      color: '#f56c6c',
    },
    {
      label: t('xai.aggregateOcclusion'),
      value: `${current.occlusion?.available_count ?? 0} / ${current.provenance?.n_explanations ?? 0}`,
      icon: 'Grid',
      color: '#e6a23c',
    },
  ]
})

const figures = computed(() => {
  const current = doc.value
  if (!current) return []
  const output = []
  const push = (asset, caption) => {
    if (asset && images.value[asset]) {
      output.push({ key: asset, src: images.value[asset], caption })
    }
  }
  for (const entry of current.intrinsic || []) {
    push(entry.topomap, `${t('xai.intrinsicColumn')} · ${tr(entry.label) || entry.id}`)
    for (const perClass of entry.per_class || []) {
      push(perClass.topomap, t('xai.figIntrinsicClass', {
        label: tr(entry.label) || entry.id,
        class: tr(perClass.label) || perClass.class,
      }))
    }
  }
  push(current.occlusion?.topomap, t('xai.figOcclusion'))
  if (current.bands?.topomap) {
    push(current.bands.topomap, t('xai.figBand', {
      band: (current.bands.topomap_band || []).join('–'),
    }))
  }
  if (current.gradient?.topomap) {
    push(current.gradient.topomap, t('xai.figGradient', {
      method: gradientLabel(current.gradient.method),
    }))
  }
  push(current.audit?.topomap, t('xai.figAudit'))
  return output
})

const channelRows = computed(() => {
  const current = doc.value
  if (!current) return []
  const channels = current.dataset?.channel_names || []
  const occlusion = current.occlusion?.channel_importance || []
  const intrinsic = (current.intrinsic || [])[0]?.values || []
  const bandKey = bandKeyOf(current.bands?.topomap_band)
  const band = current.bands?.channel_importance?.[bandKey] || []
  const gradient = current.gradient?.channel_magnitude || []
  const power = current.audit?.channel_importance || []
  return channels
    .map((name, index) => ({
      name,
      occlusion: occlusion[index],
      intrinsic: intrinsic[index],
      band: band[index],
      gradient: gradient[index],
      power: power[index],
    }))
    .sort((left, right) => Math.abs(right.occlusion ?? 0) - Math.abs(left.occlusion ?? 0))
})

const heatmapData = computed(() => {
  const occlusion = doc.value?.occlusion
  if (!occlusion?.matrix?.length || !occlusion?.window_times?.length) return null
  const groups = occlusion.groups || []
  const labels = groups.map((group, index) => {
    const names = group.map((position) => doc.value.dataset?.channel_names?.[position] || `#${position}`)
    return names.length > 3 ? `G${index + 1}` : names.join('+')
  })
  const times = occlusion.window_times.map(([start, stop]) => (Number(start) + Number(stop)) / 2)
  const values = []
  let extent = 0
  occlusion.matrix.forEach((row, groupIndex) => {
    row.forEach((value, windowIndex) => {
      const numeric = Number(value) || 0
      extent = Math.max(extent, Math.abs(numeric))
      values.push([windowIndex, groupIndex, numeric])
    })
  })
  return { labels, times, values, extent: extent || 1, nWindows: times.length }
})

const heatmapHeight = computed(() => {
  const groups = doc.value?.occlusion?.groups?.length || 4
  return `${Math.max(220, Math.min(560, groups * 28 + 90))}px`
})

const heatmapWindowLabel = computed(() => {
  const occlusion = doc.value?.occlusion
  if (!occlusion) return ''
  const windows = occlusion.window_times || []
  const first = windows[0]?.[0]
  const last = windows[windows.length - 1]?.[1]
  return first === undefined ? '' : `${formatMetric(first, 2)}–${formatMetric(last, 2)} s · ${occlusion.n_windows} × ${occlusion.groups?.length || 0}`
})

const agreementLines = computed(() => {
  const faith = doc.value?.faithfulness || {}
  const lines = []
  const intrinsic = faith.agreement?.intrinsic?.spearman
  if (intrinsic !== undefined && intrinsic !== null) {
    lines.push({ label: t('xai.intrinsicColumn'), value: `ρ = ${formatMetric(intrinsic)}` })
  }
  const gradient = faith.agreement?.gradient?.spearman
  if (gradient !== undefined && gradient !== null) {
    lines.push({ label: t('xai.gradientColumn'), value: `ρ = ${formatMetric(gradient)}` })
  }
  const audit = doc.value?.audit?.channel_agreement
  if (audit !== undefined && audit !== null) {
    lines.push({ label: t('xai.bandPowerColumn'), value: `ρ = ${formatMetric(audit)}` })
  }
  if (!lines.length) lines.push({ label: t('xai.agreement'), value: t('xai.agreementEmpty') })
  lines.push({
    label: t('xai.stability'),
    value: formatMetric(faith.stability?.half_split_spearman),
  })
  lines.push({
    label: t('xai.subjectConsistency'),
    value: formatMetric(faith.subject_consistency?.spearman),
  })
  return lines
})

const sanityLine = computed(() => {
  const sanity = doc.value?.sanity
  if (!sanity) return '-'
  if (!sanity.available) return t('xai.sanityUnavailable', { reason: tr(sanity.reason) || '-' })
  return t('xai.sanityAvailable', { rho: formatMetric(sanity.spearman) })
})

const provenanceLine = computed(() => {
  const provenance = doc.value?.provenance
  if (!provenance) return '-'
  return `${t('xai.provenanceBackground', {
    n: provenance.n_background_trials,
    nTest: provenance.n_test_trials,
  })} · ${t('xai.provenanceMethods', { methods: (provenance.methods || []).join(', ') })}`
})

function supportsParadigm(item, paradigm) {
  const label = item?.paradigm || 'MI'
  const supported = label.split('/').map((part) => part.trim().toUpperCase()).filter(Boolean)
  return supported.includes(String(paradigm || 'MI').toUpperCase())
}

function statusMeta(status) {
  const meta = JOB_STATUS[status] || EXPLANATION_STATUS[status]
  if (!meta) return { label: status || '-', type: 'info' }
  return { label: t(meta.labelKey), type: meta.type }
}

function gradientLabel(method) {
  if (method === 'integrated_gradients') return t('xai.gradientIg')
  if (method === 'deeplift') return t('xai.gradientDeeplift')
  return method || '-'
}

function bandKeyOf(band) {
  if (!Array.isArray(band) || band.length !== 2) return ''
  return band.map((value) => `${Number(value)}`).join('-')
}

function formatValue(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return Number(value).toFixed(4)
}

function formatSummary(summary) {
  if (!summary || summary.mean === null || summary.mean === undefined) return '—'
  const std = summary.std === null || summary.std === undefined ? null : Number(summary.std)
  return std === null
    ? formatMetric(summary.mean, 3)
    : `${Number(summary.mean).toFixed(3)} ± ${std.toFixed(3)} (n=${summary.n})`
}

function defaultPipeline(paradigm) {
  if (paradigm === 'P300') return 'xdawn_lda'
  if (paradigm === 'SSVEP') return 'ssvep_fbcca'
  return 'csp_lda'
}

function syncDataset() {
  const dataset = current.value
  if (!dataset) return
  const options = subjectChoices(dataset)
  if (!options.includes(subjectId.value)) subjectId.value = options[0] || ''
  if (!compatiblePipelines.value.some((item) => item.id === form.pipeline)) {
    form.pipeline = defaultPipeline(dataset.paradigm)
  }
  if (!compatiblePipelines.value.some((item) => item.id === form.pipeline)) {
    form.pipeline = compatiblePipelines.value[0]?.id || ''
  }
  form.fold = Math.min(form.fold, form.folds - 1)
  form.repeat = Math.min(form.repeat, form.repeats - 1)
}

function onDatasetChange(datasetId) {
  setDataset(datasetId)
  syncDataset()
}

function onExperimentChange(experimentId) {
  const experiment = experiments.value.find((item) => item.id === experimentId)
  if (!experiment || experiment.protocol !== 'within_subject') return
  const params = experiment.protocol_params || {}
  if (params.folds) form.folds = Number(params.folds)
  if (params.repeats) form.repeats = Number(params.repeats)
  if (Array.isArray(experiment.seeds) && experiment.seeds.length) {
    form.seedText = String(experiment.seeds[0])
  }
  form.fold = Math.min(form.fold, form.folds - 1)
  form.repeat = Math.min(form.repeat, form.repeats - 1)
  form.experimentId = experimentId
}

function buildPayload() {
  const seeds = form.seedText
    .split(',')
    .map((item) => parseInt(item.trim(), 10))
    .filter((value) => !Number.isNaN(value))
  const options = {}
  if (isSynthetic.value) options.n_trials_per_class = form.trialsPerClass
  return {
    dataset_id: selection.datasetId,
    subject_ids: subjectId.value ? [subjectId.value] : [],
    classes: current.value?.classes || 2,
    options,
    protocol: 'within_subject',
    folds: form.folds,
    repeats: form.repeats,
    seeds: seeds.length ? seeds : [0],
    pipeline: form.pipeline,
    experiment_id: form.experimentId || null,
    fold: form.fold,
    repeat: form.repeat,
    window_ms: form.windowMs,
    max_blocks: form.maxBlocks,
    gradient_method: form.gradientMethod,
    include_audit: form.audit,
    include_sanity: form.sanity,
  }
}

function validate() {
  if (!selection.datasetId || !form.pipeline) {
    ElMessage.warning(t('xai.missingDataset'))
    return false
  }
  if (!subjectId.value) {
    ElMessage.warning(t('xai.subjectRequired'))
    return false
  }
  return true
}

async function run() {
  if (!validate()) return
  submitting.value = true
  try {
    const submitted = await submitExplanationJob(buildPayload())
    job.value = submitted
    ElMessage.success(t('xai.jobSubmitted'))
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
    if (latest.status === 'completed' && latest.result?.explanation_id) {
      await openExplanation(latest.result.explanation_id)
      await loadHistory()
    } else if (latest.status === 'failed') {
      ElMessage.error(jobErrorMessage(latest, t('xai.jobFailed')))
    }
  } catch {
    // the response interceptor already reported the error
  }
}

async function cancelActiveJob() {
  if (!job.value) return
  try {
    await cancelJob(job.value.id)
    job.value = { ...job.value, cancel_requested: true, message: t('xai.cancelRequestedMessage') }
    ElMessage.info(t('xai.cancelRequested'))
    schedulePoll(600)
  } catch {
    // interceptor already reported the reason
  }
}

function disposeCharts() {
  curvesCtl?.dispose()
  bandsCtl?.dispose()
  heatCtl?.dispose()
  curvesCtl = null
  bandsCtl = null
  heatCtl = null
}

async function openExplanation(id) {
  const detail = await getExplanation(id)
  explanation.value = detail
  doc.value = detail.document
  images.value = detail.images || {}
  historyVisible.value = false
  disposeCharts()
  await nextTick()
  renderCharts()
}

function exportDocument() {
  if (!doc.value) return
  downloadJson(`explanation-${explanation.value.id}.json`, doc.value)
}

function openReport() {
  if (!explanation.value?.experiment_id) return
  router.push({ path: '/report', query: { experiment: explanation.value.experiment_id } })
}

function renderCharts() {
  // The tables and figures render from the same document; charts need the
  // container to exist, so this runs after nextTick.
  const current = doc.value
  if (!current) return
  if (curvesEl.value && curvesAvailable.value) {
    if (!curvesCtl) curvesCtl = createChart(curvesEl.value)
    const deletion = current.faithfulness.deletion_curve
    const insertion = current.faithfulness.insertion_curve
    curvesCtl.chart.setOption({
      tooltip: { trigger: 'axis' },
      legend: { top: 0, data: [t('xai.deletion'), t('xai.insertion'), t('xai.randomDeletion')] },
      grid: { left: 12, right: 16, top: 44, bottom: 16, containLabel: true },
      xAxis: {
        type: 'category',
        data: deletion.fraction.map((value) => Number(value).toFixed(2)),
        name: t('xai.fraction'),
        nameLocation: 'middle',
        nameGap: 26,
        axisLabel: { color: '#606266' },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 1,
        name: t('xai.retained'),
        axisLabel: { color: '#909399' },
        splitLine: { lineStyle: { type: 'dashed' } },
      },
      series: [
        {
          name: t('xai.deletion'), type: 'line', smooth: true, symbolSize: 5,
          data: deletion.retained.map((value) => (value === null ? null : +Number(value).toFixed(4))),
          lineStyle: { color: '#f56c6c', width: 2 }, itemStyle: { color: '#f56c6c' },
        },
        {
          name: t('xai.insertion'), type: 'line', smooth: true, symbolSize: 5,
          data: insertion.retained.map((value) => (value === null ? null : +Number(value).toFixed(4))),
          lineStyle: { color: '#67c23a', width: 2 }, itemStyle: { color: '#67c23a' },
        },
        {
          name: t('xai.randomDeletion'), type: 'line', smooth: true, symbolSize: 5,
          data: deletion.random.map((value) => (value === null ? null : +Number(value).toFixed(4))),
          lineStyle: { color: '#909399', width: 2, type: 'dashed' }, itemStyle: { color: '#909399' },
        },
      ],
    }, true)
  }
  if (bandsEl.value && bandsAvailable.value) {
    if (!bandsCtl) bandsCtl = createChart(bandsEl.value)
    const bands = current.bands
    bandsCtl.chart.setOption({
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: 12, right: 16, top: 24, bottom: 16, containLabel: true },
      xAxis: {
        type: 'category',
        data: bands.bands.map(([low, high]) => `${low}–${high}`),
        axisLabel: { color: '#606266' },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        name: t('xai.bandDrop'),
        axisLabel: { color: '#909399' },
        splitLine: { lineStyle: { type: 'dashed' } },
      },
      series: [{
        type: 'bar',
        barMaxWidth: 30,
        data: bands.drop.map((value) => +Number(value).toFixed(4)),
        itemStyle: { color: '#409eff', borderRadius: [3, 3, 0, 0] },
      }],
    }, true)
  }
  if (heatEl.value && heatmapData.value) {
    if (!heatCtl) heatCtl = createChart(heatEl.value)
    const heat = heatmapData.value
    heatCtl.chart.setOption({
      tooltip: {
        position: 'top',
        formatter: (params) => {
          const [x, y, value] = params.value
          return `${heat.times[x].toFixed(2)} s · ${heat.labels[y]}<br/>Δ ${Number(value).toFixed(4)}`
        },
      },
      grid: { left: 12, right: 16, top: 20, bottom: 20, containLabel: true },
      xAxis: {
        type: 'category',
        data: heat.times.map((value) => value.toFixed(2)),
        name: 's',
        nameLocation: 'end',
        axisLabel: { color: '#606266' },
      },
      yAxis: {
        type: 'category',
        data: heat.labels,
        axisLabel: { color: '#606266', fontSize: 11 },
      },
      visualMap: {
        min: -heat.extent,
        max: heat.extent,
        calculable: false,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        inRange: { color: ['#2f6fdb', '#eaf1fb', '#d64545'] },
        textStyle: { color: '#909399', fontSize: 10 },
      },
      series: [{
        type: 'heatmap',
        data: heat.values,
        emphasis: { itemStyle: { borderColor: '#333', borderWidth: 1 } },
      }],
    }, true)
  }
}

async function loadHistory() {
  historyLoading.value = true
  try {
    const response = await getExplanations({
      experiment_id: historyExperiment.value || undefined,
      limit: 100,
    })
    history.value = response.items
  } finally {
    historyLoading.value = false
  }
}

async function openHistory() {
  historyVisible.value = true
  await loadHistory()
}

async function loadAggregate() {
  if (!form.pipeline || !selection.datasetId) return
  aggregateLoading.value = true
  aggregateError.value = ''
  try {
    aggregate.value = await getExplanationAggregate({
      dataset_id: selection.datasetId,
      pipeline: form.pipeline,
      experiment_id: aggregateExperiment.value || undefined,
      limit: 200,
    })
  } catch (error) {
    aggregate.value = null
    const detail = error?.response?.data?.detail
    aggregateError.value = typeof detail === 'string' ? detail : t('xai.aggregateEmpty')
  } finally {
    aggregateLoading.value = false
  }
}

async function openAggregate() {
  aggregateVisible.value = true
  await loadAggregate()
}

function exportAggregate() {
  if (!aggregate.value) return
  downloadJson(`explanation-aggregate-${form.pipeline || 'pipeline'}.json`, aggregate.value)
}

async function removeExplanation(row) {
  try {
    await ElMessageBox.confirm(
      t('xai.deleteConfirm', { id: row.id }),
      t('xai.delete'),
      { type: 'warning', confirmButtonText: t('common.delete'), cancelButtonText: t('common.cancel') },
    )
  } catch {
    return
  }
  await deleteExplanation(row.id)
  ElMessage.success(t('xai.deleted'))
  if (explanation.value?.id === row.id) {
    explanation.value = null
    doc.value = null
    images.value = {}
    disposeCharts()
  }
  await loadHistory()
}

onMounted(async () => {
  const [datasetResponse, pipelineResponse, experimentResponse] = await Promise.all([
    getDatasets(),
    getPipelines(),
    getExperiments({ limit: 50 }),
  ])
  datasets.value = datasetResponse.items
  pipelines.value = pipelineResponse.items
  experiments.value = experimentResponse.items
  if (!datasets.value.some((item) => item.id === selection.datasetId)) {
    setDataset(datasets.value[0]?.id || 'synthetic')
  }
  syncDataset()
  const requestedExperiment = Number(route.query.experiment)
  if (Number.isFinite(requestedExperiment) && requestedExperiment > 0) {
    form.experimentId = requestedExperiment
  }
  await loadHistory()
  const requestedExplanation = Number(route.query.explanation)
  if (Number.isFinite(requestedExplanation) && requestedExplanation > 0) {
    await openExplanation(requestedExplanation)
  }
})

onBeforeUnmount(() => {
  stopPolling()
  disposeCharts()
})
</script>

<style scoped>
.job-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; }
.job-title { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.job-pipeline { font-size: 12px; }
.job-actions { display: flex; gap: 8px; }
.job-message { margin: 8px 0 10px; color: #606266; font-size: 13px; }
.result-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; align-items: center; }
.result-note { font-size: 12px; }
.panel-col { margin-bottom: 16px; }
.section-head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; }
.section-title { font-size: 13px; font-weight: 600; margin-bottom: 8px; }
.stat-note { font-size: 12px; line-height: 1.8; color: #606266; margin: 10px 0 0; }
.note-list { margin: 4px 0 0; padding-left: 18px; color: #606266; font-size: 12px; line-height: 1.9; }
.figure-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; }
.xai-fig { margin: 0; padding: 10px; border: 1px solid #e3eaf3; border-radius: 10px; background: #fbfdff; text-align: center; }
.xai-fig img { width: 100%; max-width: 230px; }
.xai-fig figcaption { margin-top: 6px; font-size: 11.5px; color: #5b6b80; }
</style>
