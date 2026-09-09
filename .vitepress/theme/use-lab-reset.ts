import { type Ref } from "vue"

// Lab inputs are JSON-compatible scalar or array values; computed outputs are not inputs.
export function useLabReset(...inputs: (Ref<unknown> | Ref<unknown>[])[]) {
	const refs = inputs.flat()
	const initial = refs.map(input => JSON.stringify(input.value))
	return () => {
		refs.forEach((input, index) => { input.value = JSON.parse(initial[index]) })
	}
}
