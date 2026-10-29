<template>
  <div class="app-container">
    <h2 class="page-title">{{ $t('dataset.title') }}</h2>
    <p class="page-desc">{{ $t('dataset.description') }}</p>

    <el-card shadow="never">
      <div class="filter-bar">
        <el-input v-model="keyword" :placeholder="$t('dataset.searchPlaceholder')" clearable style="width:260px" prefix-icon="Search" />
        <el-select v-model="paradigm" :placeholder="$t('dataset.paradigm')" clearable style="width:140px">
          <el-option v-for="p in ['MI', 'P300', 'SSVEP']" :key="p" :label="p" :value="p" />
        </el-select>
        <el-button type="primary" icon="Upload" @click="openImport">{{ $t('dataset.import') }}</el-button>
        <el-button icon="Refresh" @click="fetchItems">{{ $t('dataset.refresh') }}</el-button>
      </div>
      <el-table :data="filtered" border stripe v-loading="loading">
        <el-table-column prop="name" :label="$t('dataset.name')" min-width="230">
          <template #default="{ row }">
            <el-link type="primary" @click="openDetail(row)">{{ tr(row.name) }}</el-link>
            <div class="text-muted" style="font-size:12px">{{ tr(row.description) }}</div>
            <div v-if="paradigmSummary(row)" class="text-muted mono" style="font-size:11px">{{ paradigmSummary(row) }}</div>
          </template>
        </el-table-column>
        <el-table-column prop="paradigm" :label="$t('dataset.paradigm')" width="80" align="center">
          <template #default="{ row }"><el-tag size="small">{{ row.paradigm }}</el-tag></template>
        </el-table-column>
        <el-table-column prop="subject_count" :label="$t('dataset.subjects')" width="70" align="center" />
        <el-table-column prop="channels" :label="$t('dataset.channels')" width="70" align="center" />
        <el-table-column prop="sfreq" :label="$t('dataset.sampleRate')" width="90" align="center" />
        <el-table-column prop="classes" :label="$t('dataset.classes')" width="70" align="center" />
        <el-table-column prop="trials" :label="$t('dataset.trials')" width="80" align="center" />
        <el-table-column prop="adapter" :label="$t('dataset.adapter')" width="100" align="center" />
        <el-table-column prop="status" :label="$t('dataset.status')" width="150" align="center">
          <template #default="{ row }">
            <el-tag size="small" :type="row.ready ? 'success' : 'warning'">{{ tr(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="$t('dataset.actions')" width="120" align="center" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" @click="openDetail(row)">{{ $t('dataset.detail') }}</el-button>
            <el-button link type="primary" @click="openDetail(row)">{{ $t('dataset.split') }}</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-drawer v-model="drawerVisible" :title="tr(detail?.name)" size="560px" @closed="onDrawerClosed">
      <template v-if="detail">
        <el-descriptions :column="1" border size="small">
          <el-descriptions-item :label="$t('dataset.paradigm')">{{ detail.paradigm }}</el-descriptions-item>
          <el-descriptions-item :label="$t('dataset.paradigmMeta')">{{ paradigmSummary(detail) || '-' }}</el-descriptions-item>
          <el-descriptions-item :label="$t('dataset.subjectChannel')">{{ detail.subject_count }} / {{ detail.channels }}</el-descriptions-item>
          <el-descriptions-item :label="$t('dataset.rateClassTrials')">{{ detail.sfreq }} Hz / {{ detail.classes }} / {{ detail.trials }}</el-descriptions-item>
          <el-descriptions-item :label="$t('dataset.adapter')">{{ detail.adapter }}</el-descriptions-item>
          <el-descriptions-item :label="$t('dataset.license')">{{ tr(detail.license) }}</el-descriptions-item>
          <el-descriptions-item :label="$t('dataset.citation')">{{ tr(detail.citation) }}</el-descriptions-item>
          <el-descriptions-item :label="$t('dataset.notes')">{{ tr(detail.description) }}</el-descriptions-item>
        </el-descriptions>

        <el-divider content-position="left">{{ $t('dataset.previewTitle') }}</el-divider>
        <el-form label-width="90px">
          <el-form-item :label="$t('dataset.previewSubjects')">
            <el-select v-model="drawerSubjects" multiple collapse-tags clearable :placeholder="$t('dataset.defaultSubjects')" style="width:100%">
              <el-option v-for="subject in subjectOptions" :key="subject" :label="subject" :value="subject" />
            </el-select>
          </el-form-item>
        </el-form>
        <el-space>
          <el-button type="primary" icon="View" :loading="previewLoading" @click="runPreview">{{ $t('dataset.loadPreview') }}</el-button>
          <el-button icon="Grid" @click="applySelection">{{ $t('dataset.toProtocol') }}</el-button>
        </el-space>

        <template v-if="preview">
          <el-alert
            style="margin:14px 0"
            type="success"
            :closable="false"
            :title="$t('dataset.previewShape', {
              trials: preview.shape.trials,
              channels: preview.shape.channels,
              times: preview.shape.times,
              signature: preview.dataset.data_signature.slice(0, 16),
            })"
          />
          <el-descriptions :column="2" border size="small">
            <el-descriptions-item :label="$t('dataset.actualSubjects')">{{ preview.subject_ids.join(', ') }}</el-descriptions-item>
            <el-descriptions-item :label="$t('dataset.session')">{{ preview.session_ids.join(', ') }}</el-descriptions-item>
            <el-descriptions-item :label="$t('dataset.paradigmMeta')">{{ preview.dataset.paradigm }} · {{ paradigmSummary(preview.dataset) || '-' }}</el-descriptions-item>
            <el-descriptions-item :label="$t('dataset.labelCounts')">
              <span class="mono">{{ formatCounts(preview.label_counts) }}</span>
            </el-descriptions-item>
            <el-descriptions-item :label="$t('dataset.labelNames')" :span="2">{{ preview.label_names.join(' / ') || '-' }}</el-descriptions-item>
          </el-descriptions>
          <el-table :data="preview.trials" size="small" border style="margin-top:14px">
            <el-table-column prop="trial_id" label="trial_id" min-width="190" />
            <el-table-column prop="subject" :label="$t('dataset.previewSubjects')" width="70" align="center" />
            <el-table-column prop="session" :label="$t('dataset.session')" width="90" align="center" />
            <el-table-column prop="label_name" :label="$t('dataset.label')" width="90" align="center" />
          </el-table>
        </template>
      </template>
    </el-drawer>

    <el-dialog v-model="importVisible" :title="$t('dataset.importTitle')" width="760px" @closed="resetImport">
      <el-steps :active="importStep" finish-status="success" align-center style="margin-bottom:20px">
        <el-step :title="$t('dataset.stepUpload')" />
        <el-step :title="importInfo?.kind === 'epochs' ? $t('dataset.stepSummary') : $t('dataset.stepMapping')" />
        <el-step :title="$t('dataset.stepMeta')" />
      </el-steps>

      <div v-if="importStep === 0">
        <el-upload
          ref="uploadRef"
          drag
          :auto-upload="false"
          :limit="1"
          accept=".npz,.edf,.fif,.csv"
          :on-change="onFileChange"
          :on-remove="onFileRemove"
          style="width:100%"
        >
          <el-icon class="el-icon--upload"><Upload /></el-icon>
          <div class="el-upload__text">{{ $t('dataset.uploadText') }}<em>{{ $t('dataset.clickSelect') }}</em></div>
          <template #tip>
            <div class="el-upload__tip">{{ $t('dataset.uploadHint') }}</div>
          </template>
        </el-upload>
        <div style="margin-top:14px">
          <el-button type="primary" icon="Search" :loading="inspectLoading" :disabled="!importFile" @click="inspectFile">
            {{ $t('dataset.parse') }}
          </el-button>
        </div>
        <el-alert
          v-if="importInfo"
          style="margin-top:14px"
          type="success"
          :closable="false"
          :title="$t('dataset.parsed', { name: importInfo.filename, size: formatBytes(importInfo.size) })"
        />
      </div>

      <div v-else-if="importStep === 1 && importInfo?.kind === 'epochs'">
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item :label="$t('dataset.format')">{{ importInfo.format }}</el-descriptions-item>
          <el-descriptions-item :label="$t('dataset.trials')">{{ importInfo.n_trials }}</el-descriptions-item>
          <el-descriptions-item :label="$t('dataset.channelsLabel')">{{ importInfo.channels.length }}</el-descriptions-item>
          <el-descriptions-item :label="$t('dataset.sampleRate')">{{ importInfo.sfreq }} Hz</el-descriptions-item>
          <el-descriptions-item :label="$t('dataset.window')">
            {{ importInfo.tmin }} – {{ importInfo.tmax }} s
          </el-descriptions-item>
          <el-descriptions-item :label="$t('dataset.subjectsList')">{{ importInfo.subjects.join(', ') }}</el-descriptions-item>
          <el-descriptions-item :label="$t('dataset.classes')" :span="2">
            {{ importInfo.label_names.join(' / ') || '-' }}
          </el-descriptions-item>
        </el-descriptions>
        <el-alert
          style="margin-top:14px"
          type="info"
          :closable="false"
          :title="$t('dataset.npzNote')"
        />
      </div>

      <div v-else-if="importStep === 1">
        <el-form v-if="importInfo" label-width="110px">
          <el-form-item :label="$t('dataset.actualParadigm')">
            <el-select v-model="importOptions.paradigm" style="width:180px" @change="onParadigmChange">
              <el-option :label="$t('dataset.paradigm') + ' MI'" value="MI" />
              <el-option label="P300" value="P300" />
              <el-option label="SSVEP" value="SSVEP" />
            </el-select>
            <span class="text-muted" style="margin-left:12px">
              {{ importInfo.kind === 'csv'
                ? $t('dataset.csvRows', { count: importInfo.n_times })
                : $t('dataset.rawDuration', { format: importInfo.format.toUpperCase(), duration: importInfo.duration.toFixed(2) }) }}
            </span>
          </el-form-item>
          <el-form-item :label="$t('dataset.channelsLabel')">
            <el-select v-model="selectedChannels" multiple filterable collapse-tags style="width:100%">
              <el-option v-for="name in importInfo.channels" :key="name" :label="name" :value="name" />
            </el-select>
          </el-form-item>
          <el-form-item v-if="importInfo.kind === 'csv'" :label="$t('dataset.labelColumn')">
            <el-input :model-value="importInfo.label_column" disabled style="width:200px" />
            <span class="text-muted" style="margin-left:12px">
              {{ importInfo.trial_column
                ? $t('dataset.splitByTrial', { column: importInfo.trial_column })
                : $t('dataset.splitByOnset') }}
            </span>
          </el-form-item>
          <el-form-item v-else :label="$t('dataset.eventSource')">
            <el-tag size="small">{{ importInfo.event_source }}</el-tag>
            <span class="text-muted" style="margin-left:12px">{{ $t('dataset.eventHint') }}</span>
          </el-form-item>
          <el-form-item :label="$t('dataset.eventMap')">
            <el-table :data="mappingRows" size="small" border max-height="240" style="width:100%">
              <el-table-column width="50" align="center">
                <template #default="{ row }"><el-checkbox v-model="row.checked" /></template>
              </el-table-column>
              <el-table-column prop="key" :label="$t('dataset.rawEvent')" min-width="150" show-overflow-tooltip />
              <el-table-column prop="count" :label="$t('dataset.count')" width="95" align="center" />
              <el-table-column :label="$t('dataset.className')" min-width="150">
                <template #default="{ row }">
                  <el-input v-model="row.label" size="small" :disabled="!row.checked" />
                </template>
              </el-table-column>
            </el-table>
          </el-form-item>
          <el-form-item :label="$t('dataset.splitWindow')">
            <el-input-number v-model="importOptions.tmin" :step="0.1" :precision="3" controls-position="right" style="width:130px" />
            <span style="margin:0 8px">{{ $t('dataset.to') }}</span>
            <el-input-number v-model="importOptions.tmax" :step="0.1" :precision="3" controls-position="right" style="width:130px" />
            <span class="text-muted" style="margin-left:10px">{{ $t('dataset.secondsFromEvent') }}</span>
          </el-form-item>
          <el-form-item :label="$t('dataset.baseline')">
            <el-switch v-model="importOptions.baseline" />
            <template v-if="importOptions.baseline">
              <el-input-number v-model="importOptions.baselineStart" :step="0.1" :precision="3" controls-position="right" style="width:120px;margin-left:12px" />
              <span style="margin:0 8px">{{ $t('dataset.to') }}</span>
              <el-input-number v-model="importOptions.baselineEnd" :step="0.1" :precision="3" controls-position="right" style="width:120px" />
              <span class="text-muted" style="margin-left:10px">s</span>
            </template>
          </el-form-item>
          <el-form-item :label="$t('dataset.resample')">
            <el-input-number v-model="importOptions.resample" :step="1" :min="1" :max="4096" controls-position="right" style="width:130px" />
            <span class="text-muted" style="margin-left:10px">{{ $t('dataset.resampleHint') }}</span>
          </el-form-item>
          <el-form-item v-if="importOptions.paradigm === 'P300'" :label="$t('dataset.positiveLabel')">
            <el-input v-model="importOptions.positiveLabel" style="width:200px" />
          </el-form-item>
          <el-form-item v-if="importOptions.paradigm === 'SSVEP'" :label="$t('dataset.stimulusFrequencies')">
            <el-input v-model="importOptions.stimulusFrequencies" :placeholder="$t('dataset.stimulusPlaceholder')" style="width:100%" />
          </el-form-item>
        </el-form>
      </div>

      <div v-else>
        <el-form label-width="110px">
          <el-form-item :label="$t('dataset.datasetId')">
            <el-input v-model="importForm.datasetId" placeholder="my-mi-2026" />
          </el-form-item>
          <el-form-item :label="$t('dataset.nameLabel')">
            <el-input v-model="importForm.name" :placeholder="$t('dataset.namePlaceholder')" />
          </el-form-item>
          <el-form-item :label="$t('dataset.license')">
            <el-input v-model="importForm.license" :placeholder="$t('dataset.licensePlaceholder')" />
          </el-form-item>
          <el-form-item :label="$t('dataset.citation')">
            <el-input v-model="importForm.citation" :placeholder="$t('dataset.citationPlaceholder')" />
          </el-form-item>
          <el-form-item :label="$t('dataset.notes')">
            <el-input v-model="importForm.description" type="textarea" :rows="2" />
          </el-form-item>
          <el-form-item :label="$t('dataset.subjectSession')">
            <el-input v-model="importOptions.subjectId" placeholder="S01" style="width:150px" />
            <span style="margin:0 10px">/</span>
            <el-input v-model="importOptions.sessionId" placeholder="0" style="width:150px" />
          </el-form-item>
          <el-form-item :label="$t('dataset.overwrite')">
            <el-switch v-model="importForm.overwrite" />
          </el-form-item>
        </el-form>
      </div>

      <template #footer>
        <el-button v-if="importStep > 0" @click="previousStep">{{ $t('common.back') }}</el-button>
        <el-button @click="importVisible = false">{{ $t('common.cancel') }}</el-button>
        <el-button v-if="importStep < 2" type="primary" :disabled="!importInfo" @click="nextStep">{{ $t('common.next') }}</el-button>
        <el-button v-else type="primary" :loading="importLoading" :disabled="!importInfo" @click="submitImport">
          {{ $t('dataset.importAndRegister') }}
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { tr } from '@/i18n/catalog'
import { useRouter } from 'vue-router'
import {
  commitDatasetImport,
  getDatasets,
  inspectDatasetImport,
  loadDataset,
} from '@/api'
import { setDataset, setSubjects, subjectChoices } from '@/utils/selection'

const { t } = useI18n()
const router = useRouter()
const items = ref([])
const keyword = ref('')
const paradigm = ref('')
const loading = ref(false)
const detail = ref(null)
const drawerVisible = ref(false)
const drawerSubjects = ref([])
const preview = ref(null)
const previewLoading = ref(false)

const importVisible = ref(false)
const importLoading = ref(false)
const inspectLoading = ref(false)
const importStep = ref(0)
const importFile = ref(null)
const importInfo = ref(null)
const mappingRows = ref([])
const selectedChannels = ref([])
const uploadRef = ref(null)
const importForm = reactive({
  datasetId: '', name: '', license: t('dataset.defaultLicense'),
  citation: '', description: '', overwrite: false,
})
const importOptions = reactive({
  tmin: 0, tmax: 1, resample: 250,
  baseline: false, baselineStart: -0.2, baselineEnd: 0,
  paradigm: 'MI', subjectId: 'S01', sessionId: '0',
  positiveLabel: 'Target', stimulusFrequencies: '',
})

const filtered = computed(() => items.value.filter((item) => {
  const text = `${item.name}${item.citation}${item.adapter}${item.description}`.toLowerCase()
  const matchKeyword = !keyword.value || text.includes(keyword.value.toLowerCase())
  return matchKeyword && (!paradigm.value || item.paradigm === paradigm.value)
}))

const subjectOptions = computed(() => (detail.value ? subjectChoices(detail.value) : []))

function formatCounts(counts) {
  return Object.entries(counts || {}).map(([key, value]) => `${key}:${value}`).join('  ')
}

function paradigmSummary(item) {
  const meta = item?.paradigm_meta || {}
  if (meta.stimulus_frequencies?.length) {
    return t('dataset.stimulusFreq', { values: meta.stimulus_frequencies.map((value) => `${value} Hz`).join(' / ') })
  }
  if (meta.positive_label) {
    return `${t('dataset.positiveLabelTag', { label: meta.positive_label })}${meta.window_seconds ? ` · ${t('dataset.windowTag', { seconds: meta.window_seconds })}` : ''}`
  }
  if (meta.window_seconds) return t('dataset.windowTag', { seconds: meta.window_seconds })
  return ''
}

async function fetchItems() {
  loading.value = true
  try {
    items.value = (await getDatasets()).items
  } finally {
    loading.value = false
  }
}

function openDetail(row) {
  detail.value = row
  drawerSubjects.value = []
  preview.value = null
  drawerVisible.value = true
}

function onDrawerClosed() {
  detail.value = null
  preview.value = null
}

async function runPreview() {
  if (!detail.value) return
  previewLoading.value = true
  try {
    preview.value = await loadDataset(detail.value.id, {
      subject_ids: drawerSubjects.value,
      classes: null,
      options: {},
    })
  } finally {
    previewLoading.value = false
  }
}

function applySelection() {
  if (!detail.value) return
  setDataset(detail.value.id)
  setSubjects(drawerSubjects.value)
  router.push('/protocol')
}

function openImport() {
  resetImport()
  importVisible.value = true
}

function formatBytes(size) {
  if (!size) return '0 B'
  if (size < 1024) return `${size} B`
  if (size < 1024 ** 2) return `${(size / 1024).toFixed(1)} KiB`
  return `${(size / 1024 ** 2).toFixed(1)} MiB`
}

function onFileChange(file) {
  importFile.value = file.raw || null
  importInfo.value = null
  mappingRows.value = []
  importStep.value = 0
}

function onFileRemove() {
  importFile.value = null
  importInfo.value = null
  mappingRows.value = []
  importStep.value = 0
}

function resetImport() {
  uploadRef.value?.clearFiles()
  importFile.value = null
  importInfo.value = null
  mappingRows.value = []
  selectedChannels.value = []
  importStep.value = 0
  importForm.datasetId = ''
  importForm.name = ''
  importForm.license = t('dataset.defaultLicense')
  importForm.citation = ''
  importForm.description = ''
  importForm.overwrite = false
  Object.assign(importOptions, {
    tmin: 0, tmax: 1, resample: 250,
    baseline: false, baselineStart: -0.2, baselineEnd: 0,
    paradigm: 'MI', subjectId: 'S01', sessionId: '0',
    positiveLabel: 'Target', stimulusFrequencies: '',
  })
}

function defaultWindow(paradigm, duration) {
  const windows = { MI: [0.5, 3.5], P300: [0, 0.8], SSVEP: [0.15, 4.15] }
  let [tmin, tmax] = windows[paradigm] || [0, 1]
  if (duration > 0) {
    if (tmin >= duration) tmin = 0
    tmax = Math.min(tmax, Math.max(tmin + 0.2, duration))
  }
  return [tmin, tmax]
}

function onParadigmChange(value) {
  const [tmin, tmax] = defaultWindow(value, importInfo.value?.duration || 0)
  importOptions.tmin = tmin
  importOptions.tmax = tmax
}

async function inspectFile() {
  if (!importFile.value) {
    ElMessage.warning(t('dataset.selectFile'))
    return
  }
  inspectLoading.value = true
  try {
    const formData = new FormData()
    formData.append('file', importFile.value)
    const info = await inspectDatasetImport(formData)
    importInfo.value = info
    selectedChannels.value = [...(info.channels || [])]
    const suggested = new Set(info.suggested_events || [])
    mappingRows.value = Object.entries(info.event_counts || {}).map(([key, count]) => ({
      key,
      count,
      checked: suggested.has(key),
      label: info.default_mapping?.[key] || key,
    }))
    Object.assign(importOptions, {
      tmin: info.defaults?.tmin ?? 0,
      tmax: info.defaults?.tmax ?? 1,
      resample: info.defaults?.resample ?? info.sfreq ?? 250,
      baseline: false,
      paradigm: info.paradigm || 'MI',
    })
    if (!importForm.name) importForm.name = info.filename.replace(/\.[^.]+$/, '')
    if (!importForm.datasetId) {
      const stem = info.filename
        .replace(/\.[^.]+$/, '')
        .toLowerCase()
        .replace(/[^a-z0-9_-]+/g, '-')
        .replace(/^-+|-+$/g, '')
        .slice(0, 63)
      if (/^[a-z0-9]/.test(stem) && stem.length >= 2) importForm.datasetId = stem
    }
    importStep.value = 1
    if (info.kind !== 'epochs' && !info.ready) {
      ElMessage.warning(t('dataset.noEvents'))
    }
  } finally {
    inspectLoading.value = false
  }
}

function previousStep() {
  if (importStep.value > 0) importStep.value -= 1
}

function nextStep() {
  if (!importInfo.value) {
    ElMessage.warning(t('dataset.fileParseFirst'))
    return
  }
  if (importStep.value === 1 && importInfo.value.kind !== 'epochs') {
    if (!selectedChannels.value.length) {
      ElMessage.warning(t('dataset.chooseChannel'))
      return
    }
    const mapped = mappingRows.value.filter((row) => row.checked && row.label.trim())
    if (mapped.length < 2) {
      ElMessage.warning(t('dataset.chooseEvents'))
      return
    }
    if (new Set(mapped.map((row) => row.label.trim())).size !== mapped.length) {
      ElMessage.warning(t('dataset.duplicateClass'))
      return
    }
    if (!(Number(importOptions.tmax) > Number(importOptions.tmin))) {
      ElMessage.warning(t('dataset.invalidWindow'))
      return
    }
  }
  importStep.value = Math.min(2, importStep.value + 1)
}

async function submitImport() {
  if (!importInfo.value) return
  if (!/^[a-z0-9][a-z0-9_-]{1,62}$/.test(importForm.datasetId)) {
    ElMessage.warning(t('dataset.invalidId'))
    return
  }
  if (importOptions.baseline && Number(importOptions.baselineStart) >= Number(importOptions.baselineEnd)) {
    ElMessage.warning(t('dataset.invalidBaseline'))
    return
  }
  importLoading.value = true
  try {
    const eventMap = Object.fromEntries(
      mappingRows.value
        .filter((row) => row.checked && row.label.trim())
        .map((row) => [row.key, row.label.trim()]),
    )
    const frequencies = importOptions.stimulusFrequencies
      .split(/[,，\s]+/)
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value) && value > 0)
    const payload = {
      token: importInfo.value.token,
      dataset_id: importForm.datasetId,
      name: importForm.name || importForm.datasetId,
      license: importForm.license,
      citation: importForm.citation,
      description: importForm.description,
      overwrite: importForm.overwrite,
      channels: selectedChannels.value,
      event_map: importInfo.value.kind === 'epochs' ? null : eventMap,
      label_column: importInfo.value.label_column || null,
      time_column: importInfo.value.time_column || null,
      trial_column: importInfo.value.trial_column || null,
      subject_column: importInfo.value.subject_column || null,
      session_column: importInfo.value.session_column || null,
      sfreq: importInfo.value.sfreq || null,
      tmin: Number(importOptions.tmin),
      tmax: Number(importOptions.tmax),
      baseline: importOptions.baseline
        ? [Number(importOptions.baselineStart), Number(importOptions.baselineEnd)]
        : null,
      resample: importOptions.resample ? Number(importOptions.resample) : null,
      paradigm: importOptions.paradigm,
      subject_id: importOptions.subjectId || 'S01',
      session_id: importOptions.sessionId || '0',
      positive_label: importOptions.positiveLabel || null,
      stimulus_frequencies: importOptions.paradigm === 'SSVEP' && frequencies.length
        ? frequencies
        : null,
    }
    const result = await commitDatasetImport(payload)
    const importedId = result.item.id
    const trials = result.summary?.n_trials
    ElMessage.success(trials
      ? t('dataset.importedTrials', { name: result.item.name, count: trials })
      : t('dataset.imported', { name: result.item.name }))
    importVisible.value = false
    await fetchItems()
    const imported = items.value.find((item) => item.id === importedId)
    if (imported) openDetail(imported)
  } finally {
    importLoading.value = false
  }
}

watch(importVisible, (value) => { if (!value) resetImport() })
onMounted(fetchItems)
</script>
