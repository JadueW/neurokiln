<template>
  <div class="layout">
    <Sidebar :collapsed="collapsed" />
    <div class="layout-main" :style="{ marginLeft: collapsed ? '64px' : '220px' }">
      <Navbar :collapsed="collapsed" @toggle="collapsed = !collapsed" @refresh="refresh" />
      <TagsView :tags="tags" :active="route.path" @select="onSelect" @close="onClose" />
      <AppMain :key="mainKey" />
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import Sidebar from './components/Sidebar.vue'
import Navbar from './components/Navbar.vue'
import TagsView from './components/TagsView.vue'
import AppMain from './components/AppMain.vue'

const route = useRoute()
const router = useRouter()
const collapsed = ref(false)
const mainKey = ref(0)
const tags = ref([])

watch(
  () => route.path,
  () => {
    if (!tags.value.some((t) => t.path === route.path)) {
      tags.value.push({ path: route.path, titleKey: route.meta.titleKey || 'menu.dashboard' })
    }
  },
  { immediate: true },
)

function onSelect(path) {
  if (path !== route.path) router.push(path)
}
function onClose(path) {
  const index = tags.value.findIndex((t) => t.path === path)
  tags.value = tags.value.filter((t) => t.path !== path)
  if (path === route.path) {
    const next = tags.value[index - 1] || tags.value[0]
    router.push(next ? next.path : '/dashboard')
  }
}
function refresh() {
  mainKey.value += 1
}
</script>

<style scoped>
.layout { height: 100%; }
.layout-main {
  height: 100%; display: flex; flex-direction: column;
  transition: margin-left 0.2s ease; overflow: hidden;
}
</style>
