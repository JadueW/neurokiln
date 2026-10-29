<template>
  <div class="app-container">
    <h2 class="page-title">{{ $t('codex.title') }}</h2>
    <p class="page-desc">{{ $t('codex.description') }}</p>
    <div class="filter-bar">
      <el-input v-model="keyword" :placeholder="$t('codex.searchPlaceholder')" clearable style="width:280px" prefix-icon="Search" />
      <el-radio-group v-model="paradigm">
        <el-radio-button value="">{{ $t('common.all') }}</el-radio-button>
        <el-radio-button value="MI">MI</el-radio-button>
        <el-radio-button value="P300">P300</el-radio-button>
        <el-radio-button value="SSVEP">SSVEP</el-radio-button>
      </el-radio-group>
    </div>
    <el-row :gutter="16">
      <el-col v-for="item in filtered" :key="item.name" :xs="24" :sm="12" :md="8" style="margin-bottom:16px">
        <el-card shadow="hover" style="height:100%">
          <template #header>
            <div style="display:flex;justify-content:space-between;align-items:center">
              <b>{{ item.name }}</b>
              <el-tag size="small" :type="item.gpu ? 'warning' : 'success'">{{ item.gpu ? 'GPU' : 'CPU' }}</el-tag>
            </div>
          </template>
          <el-space wrap style="margin-bottom:10px">
            <el-tag size="small" effect="plain">{{ tr(item.type) }}</el-tag>
            <el-tag size="small" effect="plain" type="info">{{ item.paradigm }}</el-tag>
            <el-tag size="small" effect="plain" type="danger">{{ $t('codex.cost', { cost: tr(item.cost) }) }}</el-tag>
          </el-space>
          <p style="font-size:13px;margin:6px 0"><b>{{ $t('codex.params') }}</b><span class="mono">{{ tr(item.params) }}</span></p>
          <p style="font-size:13px;margin:6px 0"><b>{{ $t('codex.pitfall') }}</b>{{ tr(item.pitfall) }}</p>
          <p class="text-muted" style="font-size:12px;margin:6px 0">{{ $t('codex.reference', { ref: item.ref }) }}</p>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { getCodex } from '@/api'
import { tr } from '@/i18n/catalog'

const { t } = useI18n()
const items = ref([])
const keyword = ref('')
const paradigm = ref('')
const filtered = computed(() => items.value.filter((item) => {
  const text = `${item.name}${item.type}${tr(item.type)}${item.pitfall}${tr(item.pitfall)}${item.ref}`.toLowerCase()
  return (!keyword.value || text.includes(keyword.value.toLowerCase()))
    && (!paradigm.value || item.paradigm.includes(paradigm.value))
}))
onMounted(async () => { items.value = (await getCodex()).items })
</script>
