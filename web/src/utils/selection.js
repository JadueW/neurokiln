import { reactive, watch } from 'vue'

const KEY = 'neurokiln-dataset-selection'

function readSaved() {
  try {
    return JSON.parse(localStorage.getItem(KEY) || '{}')
  } catch {
    return {}
  }
}

const saved = readSaved()

export const selection = reactive({
  datasetId: saved.datasetId || 'synthetic',
  subjectIds: Array.isArray(saved.subjectIds) ? saved.subjectIds : [],
})

watch(
  selection,
  (value) => localStorage.setItem(KEY, JSON.stringify(value)),
  { deep: true },
)

export function setDataset(datasetId) {
  if (selection.datasetId !== datasetId) {
    selection.datasetId = datasetId
    selection.subjectIds = []
  }
}

export function setSubjects(subjectIds) {
  selection.subjectIds = [...subjectIds]
}

export function subjectChoices(dataset) {
  if (!dataset) return []
  if (dataset.subject_ids?.length) return dataset.subject_ids
  if (dataset.adapter === 'builtin') {
    return Array.from({ length: 9 }, (_, index) => `S${String(index + 1).padStart(2, '0')}`)
  }
  if (dataset.adapter === 'moabb') {
    return Array.from(
      { length: dataset.subjects || 1 },
      (_, index) => `S${String(index + 1).padStart(2, '0')}`,
    )
  }
  return [`S${String(1).padStart(2, '0')}`]
}

export function selectionPayload(extra = {}) {
  return {
    dataset_id: selection.datasetId,
    subject_ids: [...selection.subjectIds],
    classes: null,
    options: {},
    ...extra,
  }
}
