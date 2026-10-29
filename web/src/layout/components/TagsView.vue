<template>
  <div class="tags-view">
    <el-scrollbar>
      <div class="tags-inner">
        <span
          v-for="tag in tags"
          :key="tag.path"
          class="tag"
          :class="{ active: tag.path === active }"
          @click="$emit('select', tag.path)"
        >
          <i v-if="tag.path === active" class="dot" />{{ $t(tag.titleKey) }}
          <el-icon v-if="tags.length > 1" class="close" @click.stop="$emit('close', tag.path)"><Close /></el-icon>
        </span>
      </div>
    </el-scrollbar>
  </div>
</template>

<script setup>
defineProps({ tags: { type: Array, default: () => [] }, active: String })
defineEmits(['select', 'close'])
</script>

<style scoped>
.tags-view {
  height: var(--tags-height); background: #fff; border-bottom: 1px solid #d8dce5;
  box-shadow: 0 1px 3px rgba(0,0,0,.04);
}
.tags-inner { display: flex; align-items: center; gap: 6px; padding: 4px 12px; height: var(--tags-height); }
.tag {
  display: inline-flex; align-items: center; gap: 5px; height: 26px; padding: 0 10px;
  border: 1px solid #d8dce5; color: #495060; font-size: 12px; cursor: pointer;
  background: #fff; white-space: nowrap;
}
.tag.active { background: #409eff; border-color: #409eff; color: #fff; }
.dot { width: 7px; height: 7px; border-radius: 50%; background: #fff; }
.close { font-size: 12px; border-radius: 50%; }
.close:hover { background: rgba(0,0,0,.16); color: #fff; }
</style>
