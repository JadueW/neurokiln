<template>
  <div class="app-container">
    <h2 class="page-title">{{ $t('protocol.title') }}</h2>
    <p class="page-desc">{{ $t('protocol.description') }}</p>

    <el-card shadow="never" style="margin-bottom:16px">
      <el-form :inline="true" label-width="92px">
        <el-form-item :label="$t('protocol.dataset')">
          <el-select :model-value="selection.datasetId" style="width:240px" @change="onDatasetChange">
            <el-option v-for="item in datasets" :key="item.id" :label="tr(item.name)" :value="item.id" :disabled="!item.ready">
              <span>{{ tr(item.name) }}</span>
              <span class="text-muted" style="float:right;font-size:12px">{{ item.adapter }}</span>
            </el-option>
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('protocol.subjects')">
          <el-select v-model="selection.subjectIds" multiple collapse-tags clearable style="width:180px" :placeholder="$t('protocol.defaultSubjects')">
            <el-option v-for="subject in subjectOptions" :key="subject" :label="subject" :value="subject" />
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('protocol.protocol')">
          <el-select v-model="form.protocol" style="width:190px" @change="result = null">
            <el-option :label="$t('protocol.withinSubject')" value="within_subject" />
            <el-option :label="$t('protocol.loso')" value="loso" />
            <el-option :label="$t('protocol.crossSession')" value="cross_session" />
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('protocol.folds')" v-if="form.protocol === 'within_subject'">
          <el-input-number v-model="form.folds" :min="2" :max="10" />
        </el-form-item>
        <el-form-item :label="$t('protocol.repeats')" v-if="form.protocol === 'within_subject'">
          <el-input-number v-model="form.repeats" :min="1" :max="5" />
        </el-form-item>
        <el-form-item :label="$t('protocol.seeds')" v-if="form.protocol === 'within_subject'">
          <el-input v-model="seedText" style="width:140px" placeholder="0,1,2" />
        </el-form-item>
        <el-form-item :label="$t('protocol.classes')" v-if="classOptions.length">
          <el-select v-model="form.classes" style="width:100px">
            <el-option v-for="count in classOptions" :key="count" :label="count" :value="count" />
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('protocol.trialsPerClass')" v-if="isSynthetic">
          <el-input-number v-model="form.trialsPerClass" :min="10" :max="200" :step="10" />
        </el-form-item>
        <el-form-item :label="$t('protocol.sessions')" v-if="isSynthetic && form.protocol === 'cross_session'">
          <el-input-number v-model="form.sessions" :min="2" :max="4" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" icon="Grid" :loading="loading" @click="preview">{{ $t('protocol.preview') }}</el-button>
        </el-form-item>
      </el-form>
      <el-alert
        v-if="current?.requires_download"
        type="info"
        :closable="false"
        :title="$t('protocol.downloadHint', { name: current.name })"
      />
    </el-card>

    <template v-if="result">
      <el-alert
        type="success"
        :closable="false"
        style="margin-bottom:16px"
        :title="$t('protocol.resultSummary', { splits: result.n_splits, trials: result.dataset.loaded_n_trials, signature: result.dataset.data_signature.slice(0, 16) })"
      />
      <el-card shadow="never" style="margin-bottom:16px">
        <el-descriptions :column="4" border size="small">
          <el-descriptions-item :label="$t('protocol.dataset')">{{ tr(result.dataset.name) }}</el-descriptions-item>
          <el-descriptions-item :label="$t('protocol.paradigm')">{{ result.dataset.paradigm }}{{ paradigmSummary(result.dataset) ? ` · ${paradigmSummary(result.dataset)}` : '' }}</el-descriptions-item>
          <el-descriptions-item :label="$t('protocol.subjects')">{{ result.dataset.subjects.join(', ') }}</el-descriptions-item>
          <el-descriptions-item :label="$t('protocol.channelsClasses')">{{ result.dataset.loaded_channels }} / {{ result.dataset.loaded_classes }}</el-descriptions-item>
          <el-descriptions-item :label="$t('protocol.sampleRate')">{{ result.dataset.loaded_sfreq }} Hz</el-descriptions-item>
        </el-descriptions>
      </el-card>
      <el-card shadow="never">
        <el-table :data="result.splits" border stripe max-height="460">
          <el-table-column prop="fold" :label="$t('protocol.fold')" width="60" align="center" />
          <el-table-column prop="repeat" :label="$t('protocol.repeats')" width="70" align="center" />
          <el-table-column prop="seed" :label="$t('protocol.seeds')" width="80" align="center" />
          <el-table-column prop="n_train" :label="$t('protocol.trainCount')" width="90" align="center" />
          <el-table-column prop="n_test" :label="$t('protocol.testCount')" width="90" align="center" />
          <el-table-column :label="$t('protocol.trainDistribution')" min-width="170">
            <template #default="{ row }"><span class="mono">{{ formatCounts(row.train_labels) }}</span></template>
          </el-table-column>
          <el-table-column :label="$t('protocol.testDistribution')" min-width="170">
            <template #default="{ row }"><span class="mono">{{ formatCounts(row.test_labels) }}</span></template>
          </el-table-column>
          <el-table-column :label="$t('protocol.testSubjectSession')" min-width="150">
            <template #default="{ row }">
              <el-tag v-if="row.test_subject" size="small" type="warning">{{ row.test_subject }}</el-tag>
              <span v-else-if="row.test_session" class="mono">{{ row.train_session }} → {{ row.test_session }}</span>
              <span v-else class="mono">{{ row.test_subjects.join(',') }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="signature" :label="$t('protocol.splitSignature')" width="140" align="center">
            <template #default="{ row }"><span class="mono">{{ row.signature }}</span></template>
          </el-table-column>
        </el-table>
      </el-card>
    </template>

    <el-card shadow="never" style="margin-top:16px">
      <template #header><b>{{ $t('protocol.guardTitle') }}</b></template>
      <ol class="guard-list">
        <li>{{ $t('protocol.guard1') }}</li>
        <li>{{ $t('protocol.guard2') }}</li>
        <li>{{ $t('protocol.guard3') }}</li>
        <li>{{ $t('protocol.guard4') }}</li>
      </ol>
    </el-card>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { tr } from '@/i18n/catalog'
import { getDatasets, previewProtocol } from '@/api'
import { selection, setDataset, subjectChoices } from '@/utils/selection'

const { t } = useI18n()
const datasets = ref([])
const form = reactive({
  protocol: 'within_subject', folds: 5, repeats: 1, classes: 2,
  trialsPerClass: 40, sessions: 2,
})
const seedText = ref('0')
const result = ref(null)
const loading = ref(false)

const current = computed(() => datasets.value.find((item) => item.id === selection.datasetId))
const subjectOptions = computed(() => subjectChoices(current.value))
const classOptions = computed(() => {
  const count = current.value?.classes || 2
  return count > 2 ? Array.from({ length: count - 1 }, (_, index) => index + 2) : []
})
const isSynthetic = computed(() => current.value?.adapter === 'builtin')

function formatCounts(counts) {
  return Object.entries(counts || {}).map(([key, value]) => `${key}:${value}`).join('  ')
}

function paradigmSummary(item) {
  const meta = item?.paradigm_meta || {}
  if (meta.stimulus_frequencies?.length) {
    return t('protocol.stimulusFrequency', {
      freqs: meta.stimulus_frequencies.map((value) => `${value} Hz`).join(' / '),
    })
  }
  if (meta.positive_label) {
    return meta.window_seconds
      ? t('protocol.positiveLabelWindow', { label: meta.positive_label, seconds: meta.window_seconds })
      : t('protocol.positiveLabel', { label: meta.positive_label })
  }
  if (meta.window_seconds) return t('protocol.window', { seconds: meta.window_seconds })
  return ''
}

function syncDefaults() {
  const dataset = current.value
  if (!dataset) return
  form.classes = dataset.classes || 2
  const options = subjectChoices(dataset)
  const valid = selection.subjectIds.filter((id) => options.includes(id))
  selection.subjectIds = valid.length ? valid : options.slice(0, 1)
}

function onDatasetChange(datasetId) {
  setDataset(datasetId)
  result.value = null
  syncDefaults()
}

function payload() {
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
  }
}

function validateSubjects() {
  if (form.protocol === 'loso' && selection.subjectIds.length < 2) {
    ElMessage.warning(t('protocol.losoMinSubjects'))
    return false
  }
  if (form.protocol !== 'loso' && selection.subjectIds.length !== 1) {
    ElMessage.warning(t('protocol.oneSubjectOnly'))
    return false
  }
  return true
}

async function preview() {
  if (!validateSubjects()) return
  loading.value = true
  try {
    result.value = await previewProtocol(payload())
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  datasets.value = (await getDatasets()).items
  if (!datasets.value.some((item) => item.id === selection.datasetId)) {
    setDataset(datasets.value[0]?.id || 'synthetic')
  }
  syncDefaults()
})
</script>
